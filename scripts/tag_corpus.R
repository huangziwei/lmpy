#!/usr/bin/env Rscript
# Classify + auto-tag harvested formulas.
#
# Reads:   corpus/harvested_raw.yaml, corpus/feature_matrix.yaml
# Writes:  corpus/wr.yaml, corpus/lme4.yaml, corpus/mgcv.yaml
#
# Classification (priority order):
#   1. fn == lmer|glmer|nlmer|lFormula|glFormula    -> lme4
#   2. fn == gam|bam|gamm|t2|te|ti|s               -> mgcv
#   3. formula string contains `s(`/`te(`/`ti(`/`t2(` as a call -> mgcv
#   4. formula string contains `|` (bar)           -> lme4
#   5. otherwise                                   -> wr
#
# Tagging uses:
#   stats::terms              for WR / FE parts
#   lme4::findbars + nobars   for lme4 RE parsing
#   mgcv::interpret.gam       for mgcv smooth.spec parsing
#
# Heuristic tags are a first pass; a human audit can refine.

suppressPackageStartupMessages({
  library(yaml)
  suppressWarnings({
    library(mgcv)
    library(lme4)
  })
})

HARVEST    <- "corpus/harvested_raw.yaml"
FM         <- "corpus/feature_matrix.yaml"
OUT_WR     <- "corpus/wr.yaml"
OUT_LME4   <- "corpus/lme4.yaml"
OUT_MGCV   <- "corpus/mgcv.yaml"

# -----------------------------------------------------------------------------
# Feature matrix -> flat vector of valid feature ids
# -----------------------------------------------------------------------------
fm <- yaml::read_yaml(FM)
valid_feat <- character(0)
for (section in c("shared", "wr", "lme4", "mgcv")) {
  for (axis in fm[[section]]) {
    for (val in axis$values) {
      valid_feat <- c(valid_feat,
                      sprintf("%s.%s.%s", section, axis$axis, val$id))
    }
  }
}

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
classify_entry <- function(e) {
  fn <- e$fn
  if (fn %in% c("lmer","glmer","nlmer","lFormula","glFormula","mkReTrms")) return("lme4")
  if (fn %in% c("gam","bam","gamm","s","te","ti","t2")) return("mgcv")
  fstr <- e$formula
  if (grepl("\\b(s|te|ti|t2)\\(", fstr)) return("mgcv")
  if (grepl("\\|", fstr)) return("lme4")
  "wr"
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
safe_formula <- function(s) tryCatch(as.formula(s, env = baseenv()),
                                     error = function(e) NULL,
                                     warning = function(w) NULL)

is_empty_sym <- function(x) is.symbol(x) && !nzchar(as.character(x))

# Reasons a harvested formula can't be turned into a fixture as-is.
# Returns NULL if executable, else a short string explaining why not.
unexecutable_reason <- function(e) {
  fs <- e$formula
  rd <- e$source_rd %||% ""

  # lme4 Covariance-class: us()/diag()/cs()/compSymm() are proposed-but-not-shipped
  # covariance wrappers. Their help page is documentation of a design, not runnable.
  if (identical(rd, "Covariance-class")) return("covariance-class proposal, not implemented")

  # mgcv / gamair vignette boilerplate where the basis / k / m is a free variable.
  # These need the surrounding script context to resolve `bs`, `k`, etc.
  if (grepl("\\bbs\\s*=\\s*[A-Za-z_][A-Za-z_0-9]*\\b(?!\\s*\\()", fs, perl = TRUE) &&
      !grepl("bs\\s*=\\s*[\"']", fs))
    return("bs= is an unresolved variable, not a string literal")
  if (grepl("\\bk\\s*=\\s*[A-Za-z_][A-Za-z_0-9.]*\\b(?!\\s*\\()", fs, perl = TRUE) &&
      !grepl("k\\s*=\\s*[0-9c(]", fs) &&
      !grepl("k\\s*=\\s*-?[0-9]", fs))
    return("k= is an unresolved variable, not a constant")
  if (grepl("\\bfx\\s*=\\s*[A-Za-z_][A-Za-z_0-9]*\\b", fs) &&
      !grepl("fx\\s*=\\s*(TRUE|FALSE|T|F)\\b", fs))
    return("fx= is an unresolved variable, not a literal")

  # Deferred mgcv bases that need extra structures (polygons, boundaries, etc).
  if (grepl("bs\\s*=\\s*[\"'](mrf|so|sf|ds)[\"']", fs))
    return("deferred basis (mrf/so/sf/ds) needs auxiliary structures")
  if (grepl("bs\\s*=\\s*c\\([^)]*[\"'](mrf|so|sf|ds)[\"'][^)]*\\)", fs))
    return("deferred basis (mrf/so/sf/ds) inside bs=c(...)")

  # xt = list(...) referencing bare variables (polygons, neighborhood lists, etc.)
  # — mgcv::interpret.gam eval()s the smooth term so the bare vars must resolve.
  if (grepl("xt\\s*=\\s*list\\(", fs))
    return("xt=list(...) references variables from vignette scope")

  NULL
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# -----------------------------------------------------------------------------
# WR / FE tagger
# -----------------------------------------------------------------------------
tag_wr_fe <- function(fs) {
  tags <- character(0)
  rhs_str <- sub("^[^~]*~\\s*", "", fs)

  # ---- dot `.` operator: terms() fails w/o data. Tag via string and bail early.
  if (grepl("(^|[^A-Za-z_0-9.])\\.(\\s*([+\\-*/:^]|\\^|$))", rhs_str, perl = TRUE)) {
    tags <- c(tags, "wr.op.dot")
    if (grepl("\\.\\s*-", rhs_str)) tags <- c(tags, "wr.op.dot_minus")
    if (grepl("\\.\\s*\\^\\s*[0-9]", rhs_str)) {
      if (grepl("\\.\\s*\\^\\s*3", rhs_str)) tags <- c(tags, "wr.op.power3")
      else                                    tags <- c(tags, "wr.op.power")
      tags <- c(tags, "wr.interaction.crossed_expand")
    }
    lhs <- sub("~.*$", "", fs)
    if (grepl("cbind\\(", lhs)) tags <- c(tags, "shared.response_shape.cbind")
    else if (grepl("\\(", lhs)) tags <- c(tags, "shared.response_shape.transformed")
    else                        tags <- c(tags, "shared.response_shape.univariate")
    tags <- c(tags, "wr.intercept.default")
    return(unique(tags))
  }

  f <- safe_formula(fs)
  if (is.null(f)) return(tags)
  tt <- tryCatch(terms(f, keep.order = TRUE), error = function(e) NULL)
  if (is.null(tt)) return(tags)

  has_intercept <- attr(tt, "intercept") == 1L
  n_labels <- length(attr(tt, "term.labels"))

  # ---- intercept ----------------------------------------------------------
  if (has_intercept) {
    if (grepl("^\\s*1\\s*$", rhs_str))           tags <- c(tags, "wr.intercept.only_intercept")
    else if (grepl("^\\s*1\\s*\\+", rhs_str))    tags <- c(tags, "wr.intercept.explicit_one")
    else                                          tags <- c(tags, "wr.intercept.default")
  } else {
    if (grepl("^\\s*0\\s*\\+", rhs_str))         tags <- c(tags, "wr.intercept.leading_zero")
    else if (grepl("\\+\\s*0\\b", rhs_str))      tags <- c(tags, "wr.intercept.plus_zero")
    else                                          tags <- c(tags, "wr.intercept.minus_one")
  }

  # ---- operators (detect on rhs string) -----------------------------------
  if (n_labels >= 2 || (has_intercept && n_labels >= 1)) tags <- c(tags, "wr.op.plus")
  if (grepl("\\*", rhs_str))                 tags <- c(tags, "wr.op.star")
  if (grepl("[^:]:[^:]", paste0(" ", rhs_str, " "))) tags <- c(tags, "wr.op.colon")
  if (grepl("/", rhs_str, fixed = TRUE))     tags <- c(tags, "wr.op.slash")
  if (grepl("\\)\\s*\\^\\s*[0-9]", rhs_str)) {
    if (grepl("\\^\\s*3", rhs_str))          tags <- c(tags, "wr.op.power3")
    else                                      tags <- c(tags, "wr.op.power")
  }
  if (grepl("%in%", rhs_str, fixed = TRUE))  tags <- c(tags, "wr.op.in_op")
  if (grepl("~\\s*\\.", fs) || grepl("^\\s*\\.", rhs_str) ||
      grepl("[^a-zA-Z_0-9]\\.\\s*[+\\-]", rhs_str))
    tags <- c(tags, "wr.op.dot")
  if (grepl("~\\s*\\..*-", fs))              tags <- c(tags, "wr.op.dot_minus")
  # minus of a term (not "- 1" / "- 0" / "-number")
  if (grepl("-\\s*[A-Za-z_]", rhs_str))       tags <- c(tags, "wr.op.minus")
  if (grepl("\\([^)]*\\+[^)]*\\)\\s*\\*", rhs_str)) tags <- c(tags, "wr.op.parens")

  # ---- transforms (per term label) ----------------------------------------
  labels <- attr(tt, "term.labels")
  for (lbl in labels) {
    if (grepl("^I\\(", lbl)) {
      if (grepl("\\^", lbl))       tags <- c(tags, "wr.transform.I_square")
      else if (grepl("\\*", lbl))  tags <- c(tags, "wr.transform.I_product")
      else if (grepl("\\+", lbl))  tags <- c(tags, "wr.transform.I_combine")
      else                          tags <- c(tags, "wr.transform.I_square")
    }
    if (grepl("^poly\\(", lbl)) {
      tags <- c(tags, if (grepl("raw\\s*=\\s*TRUE", lbl)) "wr.transform.poly_raw"
                      else                                 "wr.transform.poly_orth")
    }
    if (grepl("^(splines::)?bs\\(",  lbl)) {
      tags <- c(tags, "wr.transform.bs")
      if (grepl("degree\\s*=", lbl)) tags <- c(tags, "wr.transform.bs_degree")
    }
    if (grepl("^(splines::)?ns\\(",  lbl)) tags <- c(tags, "wr.transform.ns")
    if (grepl("^log1p\\(",  lbl)) tags <- c(tags, "wr.transform.log1p")
    else if (grepl("^log\\(",  lbl)) tags <- c(tags, "wr.transform.log")
    if (grepl("^sqrt\\(", lbl)) tags <- c(tags, "wr.transform.sqrt")
    if (grepl("^exp\\(",  lbl)) tags <- c(tags, "wr.transform.exp")
    if (grepl("^(sin|cos)\\(", lbl)) tags <- c(tags, "wr.transform.sin_cos")
    if (grepl("^offset\\(", lbl)) tags <- c(tags, "wr.transform.offset")
    if (grepl("^scale\\(", lbl))  tags <- c(tags, "wr.transform.scale")
    if (grepl("^cut\\(", lbl))    tags <- c(tags, "wr.transform.cut")
    if (grepl("^factor\\(", lbl)) tags <- c(tags, "wr.transform.asis_factor")
    if (grepl("^relevel\\(", lbl)) tags <- c(tags, "wr.transform.reorder_factor")
    if (grepl("^C\\(", lbl)) tags <- c(tags, "wr.contrast.C_wrapper")
  }
  # identity transform = any bare variable term
  if (any(grepl("^[A-Za-z_][A-Za-z_0-9.]*$", labels))) {
    tags <- c(tags, "wr.transform.identity")
  }

  # ---- interactions (from term labels using :) ----------------------------
  for (lbl in labels) {
    if (grepl(":", lbl, fixed = TRUE)) {
      pieces <- strsplit(lbl, ":", fixed = TRUE)[[1]]
      n <- length(pieces)
      if (n == 2) tags <- c(tags, "wr.interaction.num_num")   # can't distinguish without data
      if (n == 3) tags <- c(tags, "wr.interaction.three_way")
      if (n >= 4) tags <- c(tags, "wr.interaction.four_way")
      # transformed term inside interaction?
      if (any(grepl("^poly\\(|^I\\(|^(splines::)?bs\\(|^log\\(", pieces))) {
        tags <- c(tags, "wr.interaction.transform_in_int")
      }
    }
  }
  # star operator in source implies crossed_expand/num_factor-style interactions
  if ("wr.op.star" %in% tags) tags <- c(tags, "wr.interaction.num_factor")
  if ("wr.op.power" %in% tags || "wr.op.power3" %in% tags) {
    tags <- c(tags, "wr.interaction.crossed_expand")
  }
  if ("wr.op.slash" %in% tags) tags <- c(tags, "wr.interaction.nested")

  # ---- response shape -----------------------------------------------------
  lhs <- sub("~.*$", "", fs)
  if (grepl("cbind\\(", lhs)) tags <- c(tags, "shared.response_shape.cbind")
  else if (grepl("^[[:space:]]*[A-Za-z_][A-Za-z_0-9.]*[[:space:]]*$", lhs))
    tags <- c(tags, "shared.response_shape.univariate")
  else if (grepl("\\(", lhs))
    tags <- c(tags, "shared.response_shape.transformed")

  # ---- edges (approximate) ------------------------------------------------
  if (grepl(":", rhs_str, fixed = TRUE) && !grepl("\\*", rhs_str) &&
      !grepl("\\+", rhs_str))
    tags <- c(tags, "wr.edge.interaction_only")
  if (n_labels == 0 && has_intercept)
    tags <- c(tags, "wr.edge.single_col_X")
  if (grepl("a\\s*\\*\\s*b\\s*-\\s*a:b", rhs_str) ||
      grepl("-\\s*[A-Za-z_][A-Za-z_0-9.]*:[A-Za-z_]", rhs_str))
    tags <- c(tags, "wr.edge.subtract_inter")
  if (grepl("\\*.*-\\s*[A-Za-z_][A-Za-z_0-9.]*(\\s|$)", rhs_str))
    tags <- c(tags, "wr.edge.subtract_main")

  unique(tags)
}

# -----------------------------------------------------------------------------
# lme4 RE tagger (operates on individual bar expressions from findbars)
# -----------------------------------------------------------------------------
tag_lme4_bar <- function(bar_expr, had_double_bar) {
  tags <- character(0)
  # bar_expr is a call of form `<lhs> | <rhs>` (or `<lhs> | <rhs>` where lhs is `0 + ...`)
  lhs <- bar_expr[[2]]
  rhs <- bar_expr[[3]]

  lhs_str <- deparse1(lhs)
  rhs_str <- deparse1(rhs)

  # ---- re_spec ------------------------------------------------------------
  if (had_double_bar) {
    tags <- c(tags, "lme4.re_spec.uncorr_slope")
  }
  if (grepl("^\\s*1\\s*$", lhs_str)) {
    tags <- c(tags, "lme4.re_spec.intercept_only")
  } else if (grepl("^\\s*0\\s*\\+", lhs_str)) {
    tags <- c(tags, "lme4.re_spec.slope_only")
  } else if (grepl("^\\s*1\\s*\\+", lhs_str)) {
    n_terms <- length(strsplit(lhs_str, "\\+")[[1]])
    if (n_terms >= 3) tags <- c(tags, "lme4.re_spec.multi_corr")
    else               tags <- c(tags, "lme4.re_spec.corr_slope_expl")
  } else {
    # e.g. `x | g` — compact correlated form
    if (grepl("\\+", lhs_str)) tags <- c(tags, "lme4.re_spec.multi_corr")
    else                        tags <- c(tags, "lme4.re_spec.corr_slope")
  }
  if (grepl("poly\\(|I\\(|log\\(|bs\\(", lhs_str))
    tags <- c(tags, "lme4.re_spec.transform_slope")

  # ---- grouping -----------------------------------------------------------
  if (grepl("/", rhs_str, fixed = TRUE)) {
    tags <- c(tags, "lme4.grouping.nested_slash")
  } else if (grepl(":", rhs_str, fixed = TRUE)) {
    tags <- c(tags, "lme4.grouping.interaction_col")
  } else {
    tags <- c(tags, "lme4.grouping.single")
  }
  if (grepl("factor\\(|as\\.factor\\(|paste\\(", rhs_str))
    tags <- c(tags, "lme4.edge_lme4.re_on_coerced")

  unique(tags)
}

tag_lme4_full <- function(fs) {
  tags <- character(0)
  f <- safe_formula(fs)
  if (is.null(f)) return(tags)

  had_double_bar <- grepl("\\|\\|", fs)

  # FE part via nobars
  nb <- tryCatch(lme4::nobars(f), error = function(e) NULL)
  if (!is.null(nb)) {
    tags <- c(tags, tag_wr_fe(deparse1(nb)))
  }

  bars <- tryCatch(lme4::findbars(f), error = function(e) NULL)
  if (!is.null(bars) && length(bars) > 0) {
    for (b in bars) tags <- c(tags, tag_lme4_bar(b, had_double_bar))
  }

  # ---- grouping axis: crossed / two_groups_same ---------------------------
  if (!is.null(bars) && length(bars) >= 2) {
    rhs_strs <- vapply(bars, function(b) deparse1(b[[3]]), character(1))
    if (length(unique(rhs_strs)) == 1) {
      tags <- c(tags, "lme4.grouping.two_groups_same")
    } else {
      tags <- c(tags, "lme4.grouping.crossed")
    }
  }

  # ---- mixing -------------------------------------------------------------
  n_bars <- if (is.null(bars)) 0 else length(bars)
  if (!is.null(nb)) {
    nb_str <- deparse1(nb)
    rhs_nb <- sub("^[^~]*~\\s*", "", nb_str)
    if (n_bars == 0)                                 tags <- c(tags, "lme4.mixing.fe_only")
    else if (grepl("^\\s*1\\s*$", rhs_nb))            tags <- c(tags, "lme4.mixing.re_only")
    else                                              tags <- c(tags, "lme4.mixing.fe_plus_re")
  }

  # ---- edge family --------------------------------------------------------
  if (had_double_bar && any(grepl("lme4.re_spec.multi_corr|lme4.re_spec.factor_slope", tags)))
    tags <- c(tags, "lme4.edge_lme4.double_bar_multi")

  unique(tags)
}

# -----------------------------------------------------------------------------
# mgcv smooth tagger
# -----------------------------------------------------------------------------
BS_CLASS_TO_ID <- list(
  tp.smooth.spec  = "mgcv.bs.tp",
  tprs.smooth.spec= "mgcv.bs.tp",
  ts.smooth.spec  = "mgcv.bs.ts",
  cr.smooth.spec  = "mgcv.bs.cr",
  cs.smooth.spec  = "mgcv.bs.cs",
  cc.smooth.spec  = "mgcv.bs.cc",
  ps.smooth.spec  = "mgcv.bs.ps",
  cp.smooth.spec  = "mgcv.bs.cp",
  re.smooth.spec  = "mgcv.bs.re",
  fs.smooth.spec  = "mgcv.bs.fs",
  gp.smooth.spec  = "mgcv.bs.gp",
  ad.smooth.spec  = "mgcv.bs.ad",
  sz.smooth.spec  = "mgcv.bs.sz",
  mrf.smooth.spec = "mgcv.bs.mrf",
  so.smooth.spec  = "mgcv.bs.so"
)

tag_mgcv_smooth <- function(sp, raw_formula_str) {
  tags <- character(0)
  cls <- class(sp)[1]
  if (cls %in% names(BS_CLASS_TO_ID)) {
    tags <- c(tags, BS_CLASS_TO_ID[[cls]])
  }

  # constructor — distinguish s / te / ti / t2
  if (cls == "tensor.smooth.spec" || cls == "t2.smooth.spec") {
    # te vs ti: need to look at formula string, since both produce tensor.smooth.spec
    term_joined <- paste(sp$term, collapse = ", ")
    # Build regex to find either ti(...) or te(...) containing these vars
    if (cls == "t2.smooth.spec") {
      tags <- c(tags, "mgcv.constructor.t2")
    } else {
      # Check whether the raw formula has `ti(` around these terms
      # naive: look for ti(<same order of vars>
      pat_ti <- paste0("ti\\(\\s*", gsub(", ", "\\s*,\\s*", term_joined), "\\b")
      if (grepl(pat_ti, raw_formula_str)) {
        tags <- c(tags, "mgcv.constructor.ti_decomp")
      } else {
        tags <- c(tags, "mgcv.constructor.te")
        if (!is.null(sp$bs) && length(sp$bs) > 1 && length(unique(sp$bs)) > 1)
          tags <- c(tags, "mgcv.constructor.te_mixed_bs")
        if (!is.null(sp$bs.dim) && length(sp$bs.dim) > 1 && length(unique(sp$bs.dim)) > 1)
          tags <- c(tags, "mgcv.constructor.te_different_k")
      }
    }
  } else {
    # s() by dim
    if (is.null(sp$dim)) {
      tags <- c(tags, "mgcv.constructor.s_1d")
    } else if (sp$dim == 1) {
      tags <- c(tags, "mgcv.constructor.s_1d")
    } else if (sp$dim == 2) {
      tags <- c(tags, "mgcv.constructor.s_iso_2d")
    } else if (sp$dim >= 3) {
      tags <- c(tags, "mgcv.constructor.s_iso_3d")
    }
  }

  # modifiers
  if (!is.null(sp$bs.dim) && any(sp$bs.dim > 0 & sp$bs.dim != -1L))
    tags <- c(tags, "mgcv.modifier.k")
  if (!is.null(sp$p.order) && !all(is.na(sp$p.order)))
    tags <- c(tags, "mgcv.modifier.m_penalty")
  if (!is.null(sp$by) && as.character(sp$by) != "NA") {
    # by=factor vs by=numeric — can't tell without data; default to by_factor tag,
    # and also consider by_numeric if id= present it suggests multiple levels
    tags <- c(tags, "mgcv.modifier.by_factor")
    if (!is.null(sp$id) && !is.na(sp$id))
      tags <- c(tags, "mgcv.modifier.by_factor_id")
  }
  if (!is.null(sp$sp) && any(!is.na(sp$sp) & sp$sp >= 0))
    tags <- c(tags, "mgcv.modifier.sp_fixed")
  if (!is.null(sp$xt) && length(sp$xt) > 0)
    tags <- c(tags, "mgcv.modifier.xt_extras")
  if (!is.null(sp$point.con) || !is.null(sp$pc))
    tags <- c(tags, "mgcv.modifier.pc_point")

  unique(tags)
}

tag_mgcv_full <- function(fs) {
  tags <- character(0)
  f <- safe_formula(fs)
  if (is.null(f)) return(tags)

  ig <- tryCatch(mgcv::interpret.gam(f), error = function(e) NULL,
                 warning = function(w) NULL)
  if (is.null(ig)) return(tags)

  # FE part
  if (!is.null(ig$pf)) {
    tags <- c(tags, tag_wr_fe(deparse1(ig$pf)))
  }

  if (length(ig$smooth.spec) > 0) {
    for (sp in ig$smooth.spec) {
      tags <- c(tags, tag_mgcv_smooth(sp, fs))
    }
  }

  # ---- mixing -------------------------------------------------------------
  n_sm <- length(ig$smooth.spec)
  pf_rhs <- if (!is.null(ig$pf)) sub("^[^~]*~\\s*", "", deparse1(ig$pf)) else ""
  has_param <- nzchar(pf_rhs) && !grepl("^\\s*1\\s*$", pf_rhs)
  has_re    <- any("mgcv.bs.re" %in% tags)

  if (n_sm > 0 && !has_param) tags <- c(tags, "mgcv.mixing.smooth_only")
  if (has_param && n_sm >= 1 && !has_re) tags <- c(tags, "mgcv.mixing.parametric_plus")
  if (n_sm >= 2 && !has_re) tags <- c(tags, "mgcv.mixing.two_smooths")
  if (n_sm >= 1 && has_re) tags <- c(tags, "mgcv.mixing.smooth_and_re")
  if (has_param && has_re) tags <- c(tags, "mgcv.mixing.parametric_smooth_re")

  unique(tags)
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
raw <- yaml::read_yaml(HARVEST)
all_entries <- list()
for (pkg in names(raw$formulas)) {
  for (e in raw$formulas[[pkg]]) {
    e$pkg <- pkg
    all_entries <- c(all_entries, list(e))
  }
}
cat(sprintf("loaded %d harvested formulas\n", length(all_entries)))

buckets <- list(wr = list(), lme4 = list(), mgcv = list())
unknown_tags_seen <- character(0)

n_unexecutable <- 0L
for (i in seq_along(all_entries)) {
  e <- all_entries[[i]]
  kind <- classify_entry(e)
  reason <- unexecutable_reason(e)
  executable <- is.null(reason)

  if (executable) {
    tags <- switch(kind,
      wr   = tag_wr_fe(e$formula),
      lme4 = tag_lme4_full(e$formula),
      mgcv = tag_mgcv_full(e$formula)
    )
  } else {
    tags <- character(0)
    n_unexecutable <- n_unexecutable + 1L
  }

  # Dataset heuristic: source_rd often matches the dataset name
  dataset_hint <- e$source_rd

  id <- sprintf("%s_%04d", kind, length(buckets[[kind]]) + 1L)

  entry <- list(
    id = id,
    formula = e$formula,
    source_pkg = e$pkg,
    source_fn  = e$fn,
    source_rd  = e$source_rd,
    dataset_hint = dataset_hint,
    executable = executable,
    features = as.list(tags),
    notes = if (is.null(reason)) "" else reason
  )
  buckets[[kind]] <- c(buckets[[kind]], list(entry))

  bad <- setdiff(tags, valid_feat)
  if (length(bad) > 0) unknown_tags_seen <- unique(c(unknown_tags_seen, bad))
}

# -----------------------------------------------------------------------------
# Write outputs
# -----------------------------------------------------------------------------
write_section <- function(path, entries, kind) {
  out <- list(
    meta = list(
      generated_by = "scripts/tag_corpus.R",
      generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
      kind = kind,
      count = length(entries)
    ),
    cases = entries
  )
  yaml::write_yaml(out, path)
  cat(sprintf("  wrote %-18s  %4d cases\n", path, length(entries)))
}

cat("\nwriting corpus splits\n")
write_section(OUT_WR,   buckets$wr,   "wr")
write_section(OUT_LME4, buckets$lme4, "lme4")
write_section(OUT_MGCV, buckets$mgcv, "mgcv")

cat(sprintf("\ntotal cases: %d  (wr=%d lme4=%d mgcv=%d)\n",
            length(all_entries),
            length(buckets$wr), length(buckets$lme4), length(buckets$mgcv)))
cat(sprintf("unexecutable (covariance-class / parametric / deferred basis): %d\n",
            n_unexecutable))

if (length(unknown_tags_seen) > 0) {
  cat(sprintf("\nWARNING: %d unknown tags emitted (not in feature_matrix.yaml):\n",
              length(unknown_tags_seen)))
  for (t in sort(unknown_tags_seen)) cat("  -", t, "\n")
}
