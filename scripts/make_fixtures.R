#!/usr/bin/env Rscript
# Generate test fixtures by running R's ground-truth pipelines on each tagged
# case and dumping the result.
#
# Reads:
#   corpus/wr.yaml, corpus/lme4.yaml, corpus/mgcv.yaml  (tagged cases)
#   datasets/<pkg>/<name>.csv                            (harvested/synthetic data)
#
# Writes:
#   tests/fixtures/<id>/meta.json           always — points to source dataset
#   tests/fixtures/<id>/X.csv               WR + lme4 fixed effects + mgcv parametric
#   tests/fixtures/<id>/X_meta.json         assign, contrasts, term-labels
#   tests/fixtures/<id>/Z.mtx               lme4 random-effect design (sparse)
#   tests/fixtures/<id>/Lambdat.mtx         lme4 template
#   tests/fixtures/<id>/theta.csv           lme4 initial variance parameters
#   tests/fixtures/<id>/re_meta.json        lme4 grouping / term structure
#   tests/fixtures/<id>/smooth_<i>_X.mtx    mgcv per-smooth design block
#   tests/fixtures/<id>/smooth_<i>_S_<j>.mtx mgcv per-smooth penalty
#   tests/fixtures/<id>/smooth_meta.json    mgcv smooth specs summary
#   tests/fixtures/manifest.json            per-id status / error messages
#
# Iteration strategy: run everything in a tryCatch; never halt on a single
# case failure. The manifest captures failures for triage.

suppressPackageStartupMessages({
  library(yaml)
  library(jsonlite)
  library(Matrix)
  library(splines)                  # for ns() / bs() inside formulas
  suppressWarnings({
    library(lme4)
    library(mgcv)
    library(MASS)                   # for helper funcs some help-formulas use
  })
})

OUT_ROOT <- "tests/fixtures"
DATA_ROOT <- "datasets"
dir.create(OUT_ROOT, showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# Load tagged cases
# -----------------------------------------------------------------------------
load_cases <- function(path, kind) {
  y <- yaml::read_yaml(path)
  lapply(y$cases, function(c) { c$kind <- kind; c })
}
load_curated <- function(path) {
  # curated cases carry their own kind (wr/lme4/mgcv) inferred from features
  y <- yaml::read_yaml(path)
  lapply(y$cases, function(c) {
    fs <- c$features
    kind <- if (any(grepl("^lme4\\.", fs))) "lme4"
            else if (any(grepl("^mgcv\\.", fs))) "mgcv"
            else "wr"
    c$kind <- kind
    c
  })
}
all_cases <- c(
  load_cases("corpus/wr.yaml",   "wr"),
  load_cases("corpus/lme4.yaml", "lme4"),
  load_cases("corpus/mgcv.yaml", "mgcv"),
  load_curated("corpus/curated.yaml")
)
exec_cases <- Filter(function(c) isTRUE(c$executable), all_cases)
cat(sprintf("loaded %d cases (%d executable)\n",
            length(all_cases), length(exec_cases)))

# -----------------------------------------------------------------------------
# Dataset catalog: (pkg, name, path, cols, nrow)
# -----------------------------------------------------------------------------
catalog <- list()
for (path in Sys.glob(file.path(DATA_ROOT, "*", "*.csv"))) {
  pkg  <- basename(dirname(path))
  name <- sub("\\.csv$", "", basename(path))
  # peek header + count rows lazily (nrow only on first match for speed)
  hdr <- tryCatch(read.csv(path, nrows = 1, check.names = FALSE),
                  error = function(e) NULL)
  if (is.null(hdr)) next
  cols <- names(hdr)
  # drop leading blank column name (artifact of `write.csv(row.names=FALSE)`
  # when the frame had row names preserved earlier). In the actual data these
  # are always character row-index columns we ignore.
  cols <- cols[nzchar(cols) & cols != "X"]
  catalog[[length(catalog) + 1L]] <- list(
    pkg = pkg, name = name, path = path, cols = cols
  )
}
cat(sprintf("dataset catalog: %d files\n", length(catalog)))

# -----------------------------------------------------------------------------
# Formula var extraction + dataset matching
# -----------------------------------------------------------------------------
# Extract required vars from a formula string. For lme4, also include bar-RHS.
# For mgcv, smooth args are ordinary vars inside s()/te()/ti()/t2() — all.vars
# handles them.
extract_vars <- function(fs, kind) {
  f <- tryCatch(as.formula(fs, env = globalenv()),
                error = function(e) NULL,
                warning = function(w) NULL)
  if (is.null(f)) return(character(0))
  vs <- tryCatch(all.vars(f), error = function(e) character(0))
  # Filter out R-builtin 'magic' vars that appear as symbols
  vs <- setdiff(vs, c("pi", "T", "F", "TRUE", "FALSE", "NA"))
  # Drop names that resolve to a function in base/loaded namespaces
  # (e.g. contr.treatment, contr.sum — picked up by all.vars inside C(f, ...))
  Filter(function(v) !exists(v, mode = "function"), vs)
}

# Is this formula a dot formula `y ~ .`?
is_dot_formula <- function(fs) {
  rhs <- sub("^[^~]*~\\s*", "", fs)
  grepl("(^|[^A-Za-z_0-9.])\\.(\\s*([+\\-*/:^]|\\^|$))", rhs, perl = TRUE)
}

# Given required vars and a preferred package, pick a dataset that contains
# all required cols. Priority:
#   1. exact pkg match + contains all cols
#   2. any catalog entry containing all cols — prefer smallest n_cols
#   3. NULL (no match)
pick_dataset <- function(need_vars, pref_pkg) {
  if (length(need_vars) == 0) return(NULL)
  candidates <- list()
  for (d in catalog) {
    if (all(need_vars %in% d$cols)) {
      candidates[[length(candidates) + 1L]] <- d
    }
  }
  if (length(candidates) == 0) return(NULL)
  # prefer candidates whose pkg matches the source pkg
  if (!is.null(pref_pkg)) {
    pref <- Filter(function(d) d$pkg == pref_pkg, candidates)
    if (length(pref) > 0) candidates <- pref
  }
  # break ties by smallest column count (most focused frame)
  ncols <- vapply(candidates, function(d) length(d$cols), integer(1))
  candidates[[which.min(ncols)]]
}

# For dot formulas we just need the LHS var. Prefer a dataset in pref_pkg that
# contains the LHS var + >= 2 other numeric cols.
pick_dataset_for_dot <- function(lhs_var, pref_pkg) {
  candidates <- list()
  for (d in catalog) {
    if (lhs_var %in% d$cols && length(d$cols) >= 3) {
      candidates[[length(candidates) + 1L]] <- d
    }
  }
  if (length(candidates) == 0) return(NULL)
  if (!is.null(pref_pkg)) {
    pref <- Filter(function(d) d$pkg == pref_pkg, candidates)
    if (length(pref) > 0) candidates <- pref
  }
  candidates[[1]]
}

# -----------------------------------------------------------------------------
# Per-case fixture generation
# -----------------------------------------------------------------------------
write_mm <- function(mat, path) {
  # Ensure Matrix::writeMM receives a sparse matrix
  if (!inherits(mat, "sparseMatrix")) mat <- Matrix::Matrix(mat, sparse = TRUE)
  Matrix::writeMM(mat, path)
}

write_json <- function(obj, path) {
  writeLines(jsonlite::toJSON(obj, auto_unbox = TRUE, pretty = TRUE,
                              na = "null", null = "null"), path)
}

# datasets/R/ mirrors R's built-in `datasets` package; other pkgs map 1:1.
pkg_subdir <- function(pkg) if (pkg == "datasets") "R" else pkg

# Re-apply factor types from a <name>.schema.json sidecar, if present.
# No-op when no schema exists (i.e., the source data.frame had no factors).
apply_dataset_schema <- function(d, pkg, name) {
  path <- file.path(DATA_ROOT, pkg_subdir(pkg), paste0(name, ".schema.json"))
  if (!file.exists(path)) return(d)
  sch <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  for (col in names(sch$factors)) {
    if (!col %in% names(d)) next
    spec <- sch$factors[[col]]
    lev <- as.character(unlist(spec$levels))
    d[[col]] <- factor(as.character(d[[col]]), levels = lev,
                       ordered = isTRUE(spec$ordered))
  }
  d
}

# ---- Data-aware tagging ----------------------------------------------------
# Given the model.frame `mf` actually used, the raw data `raw`, and the fixed
# design matrix `X`, emit the shared.* feature tags that can only be known by
# looking at the data — var_type, factor_cardinality, na_policy, rank.
data_aware_tags <- function(mf, raw, X) {
  tags <- character(0)

  # var_type — one tag per type of column present in mf
  type_map <- c(numeric = "numeric", double = "numeric", integer = "integer",
                logical = "logical", character = "character",
                factor = "factor", ordered = "ordered", Date = "date",
                POSIXct = "date", POSIXlt = "date")
  for (col in names(mf)) {
    v <- mf[[col]]
    # matrix columns (from poly/ns/bs/...) are also is.numeric, so check first
    if (is.matrix(v))        tags <- c(tags, "shared.var_type.matrix_column")
    else if (is.ordered(v))  tags <- c(tags, "shared.var_type.ordered")
    else if (is.factor(v))   tags <- c(tags, "shared.var_type.factor")
    else if (is.logical(v))  tags <- c(tags, "shared.var_type.logical")
    else if (is.integer(v))  tags <- c(tags, "shared.var_type.integer")
    else if (is.numeric(v))  tags <- c(tags, "shared.var_type.numeric")
    else if (is.character(v)) tags <- c(tags, "shared.var_type.character")
    else if (inherits(v, c("Date", "POSIXct", "POSIXlt")))
      tags <- c(tags, "shared.var_type.date")
  }

  # factor_cardinality — per factor column in mf
  for (col in names(mf)) {
    v <- mf[[col]]
    if (!is.factor(v)) next
    used <- unique(as.character(v))
    declared <- levels(v)
    n_used <- length(used)
    if (n_used == 2)                tags <- c(tags, "shared.factor_cardinality.two_level")
    if (n_used >= 3 && n_used <= 5) tags <- c(tags, "shared.factor_cardinality.balanced_3_5")
    if (n_used >= 20)               tags <- c(tags, "shared.factor_cardinality.many_level")
    # singleton: any USED level appearing exactly once
    counts <- table(as.character(v))
    if (any(counts == 1))           tags <- c(tags, "shared.factor_cardinality.singleton")
    # unused_level: declared levels > used levels
    if (length(declared) > n_used)  tags <- c(tags, "shared.factor_cardinality.unused_level")
  }

  # na_policy: default model.frame uses na.omit. If raw had NAs in any var we
  # used, mf will have fewer rows — that's the omit path.
  used_vars <- names(mf)
  raw_cols <- intersect(used_vars, names(raw))
  raw_sub  <- raw[, raw_cols, drop = FALSE]
  had_na   <- any(vapply(raw_sub, function(c) any(is.na(c)), logical(1)))
  if (had_na && nrow(mf) < nrow(raw)) tags <- c(tags, "shared.na_policy.omit")

  # rank — qr decomposition of X
  if (!is.null(X) && ncol(X) > 0 && nrow(X) > 0) {
    rk <- tryCatch(qr(X)$rank, error = function(e) NA_integer_)
    if (!is.na(rk)) {
      if (rk == ncol(X))       tags <- c(tags, "shared.rank.full")
      else if (rk <  ncol(X))  tags <- c(tags, "shared.rank.aliased_cols")
      if (ncol(X) >= nrow(X))  tags <- c(tags, "shared.rank.perfect_fit")
    }
  }

  # empty_cells — any pair of factors in mf with a 0-count cross-tab cell
  fcols <- names(mf)[vapply(mf, is.factor, logical(1))]
  if (length(fcols) >= 2) {
    done <- FALSE
    for (i in seq_len(length(fcols) - 1)) {
      for (j in seq(i + 1, length(fcols))) {
        tab <- table(mf[[fcols[i]]], mf[[fcols[j]]])
        if (any(tab == 0)) {
          tags <- c(tags, "shared.rank.empty_cells"); done <- TRUE; break
        }
      }
      if (done) break
    }
  }

  unique(tags)
}

# -- WR: run model.matrix and record contrasts / assign / term.labels ---------
make_wr_fixture <- function(case, data, out_dir) {
  f <- as.formula(case$formula, env = globalenv())
  mf <- model.frame(f, data)
  X  <- model.matrix(f, mf)

  write.csv(X, file.path(out_dir, "X.csv"),
            row.names = FALSE, na = "NA")
  write_json(list(
    colnames  = colnames(X),
    assign    = attr(X, "assign"),
    contrasts = attr(X, "contrasts"),
    term_labels = attr(terms(mf), "term.labels"),
    intercept = attr(terms(mf), "intercept") == 1L,
    response  = attr(terms(mf), "response")
  ), file.path(out_dir, "X_meta.json"))

  list(rows = nrow(mf), cols_X = ncol(X),
       enriched_tags = data_aware_tags(mf, data, X))
}

# -- lme4: use lFormula to get X, Z, Lambdat, theta --------------------------
make_lme4_fixture <- function(case, data, out_dir) {
  f <- as.formula(case$formula, env = globalenv())
  # lFormula builds everything we need without running the optimizer
  lf <- lme4::lFormula(f, data = data, control = lme4::lmerControl(
    check.nobs.vs.nlev     = "ignore",
    check.nobs.vs.nRE      = "ignore",
    check.nobs.vs.rankZ    = "ignore",
    check.rankX            = "silent.drop.cols"
  ))
  X  <- lf$X
  Zt <- lf$reTrms$Zt        # transposed RE design (q x n)
  Z  <- Matrix::t(Zt)
  Lambdat <- lf$reTrms$Lambdat
  theta   <- lf$reTrms$theta
  flist   <- lf$reTrms$flist
  cnms    <- lf$reTrms$cnms  # names per grouping

  # Data frame (with response) used by lFormula — rebuild from fr
  mf <- lf$fr

  write.csv(X, file.path(out_dir, "X.csv"),
            row.names = FALSE, na = "NA")
  write_json(list(
    colnames = colnames(X),
    assign   = attr(X, "assign"),
    contrasts = attr(X, "contrasts")
  ), file.path(out_dir, "X_meta.json"))

  write_mm(Z, file.path(out_dir, "Z.mtx"))
  write_mm(Lambdat, file.path(out_dir, "Lambdat.mtx"))
  write.csv(data.frame(theta = theta), file.path(out_dir, "theta.csv"),
            row.names = FALSE)

  write_json(list(
    flist_names = names(flist),
    flist_levels = lapply(flist, function(x) levels(x)),
    cnms = cnms,
    Gp = lf$reTrms$Gp,
    nl = sapply(flist, nlevels),
    n  = nrow(mf), q = nrow(Zt)
  ), file.path(out_dir, "re_meta.json"))

  list(rows = nrow(mf), cols_X = ncol(X), q = nrow(Zt),
       n_theta = length(theta),
       enriched_tags = data_aware_tags(mf, data, X))
}

# -- mgcv: interpret.gam + per-smooth smoothCon -------------------------------
make_mgcv_fixture <- function(case, data, out_dir) {
  f <- as.formula(case$formula, env = globalenv())
  ig <- mgcv::interpret.gam(f)

  # parametric block
  pf <- ig$pf
  mf <- model.frame(pf, data)
  Xp <- model.matrix(pf, mf)
  write.csv(Xp, file.path(out_dir, "X.csv"),
            row.names = FALSE, na = "NA")
  write_json(list(
    colnames = colnames(Xp),
    assign   = attr(Xp, "assign"),
    contrasts = attr(Xp, "contrasts"),
    parametric_rhs = deparse1(pf[[3]])
  ), file.path(out_dir, "X_meta.json"))

  # per-smooth construction
  sm_summaries <- list()
  # Use full data (not just mf) since smoothCon may need vars outside the
  # parametric part
  full_data <- na.omit(data[, unique(c(all.vars(f))), drop = FALSE])
  for (i in seq_along(ig$smooth.spec)) {
    sp <- ig$smooth.spec[[i]]
    sm <- tryCatch(
      mgcv::smoothCon(sp, data = full_data, knots = NULL,
                      absorb.cons = TRUE, scale.penalty = TRUE),
      error = function(e) NULL
    )
    if (is.null(sm) || length(sm) == 0) {
      sm_summaries[[i]] <- list(index = i, class = class(sp)[1], error = "smoothCon failed")
      next
    }
    # smoothCon returns a list (usually length 1; >1 for 'by' with id)
    for (k in seq_along(sm)) {
      s <- sm[[k]]
      prefix <- sprintf("smooth_%d_%d", i, k)
      write_mm(Matrix::Matrix(s$X, sparse = TRUE),
               file.path(out_dir, paste0(prefix, "_X.mtx")))
      for (j in seq_along(s$S)) {
        write_mm(Matrix::Matrix(s$S[[j]], sparse = TRUE),
                 file.path(out_dir, sprintf("%s_S_%d.mtx", prefix, j)))
      }
    }
    sm_summaries[[i]] <- list(
      index      = i,
      class      = class(sp)[1],
      label      = sp$label,
      term       = sp$term,
      dim        = sp$dim,
      bs         = if (is.null(sp$bs)) NA_character_ else sp$bs,
      bs_dim     = sp$bs.dim,
      by         = as.character(sp$by),
      n_blocks   = length(sm),
      n_penalties = length(sm[[1]]$S)
    )
  }
  write_json(list(
    smooths = sm_summaries,
    parametric_rhs = deparse1(pf[[3]])
  ), file.path(out_dir, "smooth_meta.json"))

  list(rows = nrow(full_data), cols_X = ncol(Xp),
       n_smooths = length(ig$smooth.spec),
       enriched_tags = data_aware_tags(full_data, data, Xp))
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
manifest <- list()
n_ok <- 0; n_no_data <- 0; n_err <- 0

for (c in exec_cases) {
  id <- c$id
  kind <- c$kind
  fs <- c$formula

  # 1. Find a dataset
  need <- extract_vars(fs, kind)
  is_dot <- is_dot_formula(fs)
  if (is_dot) {
    lhs_match <- regmatches(fs, regexpr("^[^~]+", fs))
    lhs <- trimws(sub("\\(.*$", "", lhs_match))  # crude: strip transforms
    ds <- pick_dataset_for_dot(lhs, c$source_pkg)
  } else {
    ds <- pick_dataset(need, c$source_pkg)
  }

  if (is.null(ds)) {
    n_no_data <- n_no_data + 1
    manifest[[length(manifest) + 1L]] <- list(
      id = id, kind = kind, status = "no_dataset",
      need_vars = need, formula = fs, source = sprintf("%s::%s", c$source_pkg, c$source_rd)
    )
    next
  }

  # 2. Load dataset (only columns we need, or all cols for dot formulas)
  data <- tryCatch({
    d <- read.csv(ds$path, check.names = FALSE, stringsAsFactors = TRUE)
    # drop unnamed row-index column if present
    d <- d[, nzchar(names(d)), drop = FALSE]
    # Re-apply factor types lost through CSV round-trip (quoted numeric
    # factor levels get re-parsed as integer by read.csv). Without this
    # step, fs/sz/by=factor smooths silently dispatch to their fallthrough
    # tp path instead of exercising the factor-aware constructor.
    apply_dataset_schema(d, ds$pkg, ds$name)
  }, error = function(e) NULL)
  if (is.null(data)) {
    n_err <- n_err + 1
    manifest[[length(manifest) + 1L]] <- list(
      id = id, kind = kind, status = "load_error",
      dataset = sprintf("%s/%s", ds$pkg, ds$name))
    next
  }

  # 3. Create output dir
  out_dir <- file.path(OUT_ROOT, id)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  # 4. Always write meta.json
  write_json(list(
    id         = id,
    kind       = kind,
    formula    = fs,
    source_pkg = c$source_pkg,
    source_fn  = c$source_fn,
    source_rd  = c$source_rd,
    dataset    = list(pkg = ds$pkg, name = ds$name),
    features   = unlist(c$features),
    need_vars  = need
  ), file.path(out_dir, "meta.json"))

  # 5. Run system-specific pipeline
  result <- tryCatch(
    switch(kind,
      wr   = make_wr_fixture(c, data, out_dir),
      lme4 = make_lme4_fixture(c, data, out_dir),
      mgcv = make_mgcv_fixture(c, data, out_dir)
    ),
    error = function(e) list(error = conditionMessage(e)),
    warning = function(w) list(warning = conditionMessage(w))
  )

  if (!is.null(result$error)) {
    n_err <- n_err + 1
    manifest[[length(manifest) + 1L]] <- list(
      id = id, kind = kind, status = "runtime_error",
      dataset = sprintf("%s/%s", ds$pkg, ds$name),
      error = result$error, formula = fs
    )
  } else {
    n_ok <- n_ok + 1
    manifest[[length(manifest) + 1L]] <- c(
      list(id = id, kind = kind, status = "ok",
           dataset = sprintf("%s/%s", ds$pkg, ds$name)),
      result
    )
  }
}

write_json(list(
  generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  n_total = length(exec_cases),
  n_ok = n_ok, n_no_data = n_no_data, n_err = n_err,
  entries = manifest
), file.path(OUT_ROOT, "manifest.json"))

cat(sprintf("\nfixture generation: %d ok, %d no_dataset, %d errors (of %d)\n",
            n_ok, n_no_data, n_err, length(exec_cases)))
