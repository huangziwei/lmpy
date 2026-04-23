#!/usr/bin/env Rscript
# Coverage audit for the tagged corpus.
#
# Reads:
#   corpus/feature_matrix.yaml    (all axes and feature ids we aspire to cover)
#   corpus/wr.yaml                (tagged WR cases)
#   corpus/lme4.yaml              (tagged lme4 cases)
#   corpus/mgcv.yaml              (tagged mgcv cases)
#
# Writes:
#   corpus/coverage.txt           (human-readable audit)
#
# Reports:
#   (a) per-section per-axis feature coverage (count + list of uncovered values)
#   (b) forced_pairs coverage (pair satisfied iff any one executable case carries both)
#   (c) dataset assignment: which cases have a usable dataset in datasets/<pkg>/,
#       which need one.

suppressPackageStartupMessages({
  library(yaml)
  library(jsonlite)
})

FM_PATH  <- "corpus/feature_matrix.yaml"
WR       <- "corpus/wr.yaml"
LME4     <- "corpus/lme4.yaml"
MGCV     <- "corpus/mgcv.yaml"
CURATED  <- "corpus/curated.yaml"
OUT      <- "corpus/coverage.txt"
DATA     <- "datasets"
MANIFEST <- "tests/fixtures/manifest.json"

fm <- yaml::read_yaml(FM_PATH)
load_cases <- function(path) if (file.exists(path)) yaml::read_yaml(path)$cases else list()
cases <- c(load_cases(WR), load_cases(LME4), load_cases(MGCV), load_cases(CURATED))
exec_cases <- Filter(function(c) isTRUE(c$executable), cases)

# Build:
#   fixture_ok_ids — ids with status=="ok" in manifest
#   enriched_map   — id -> character vector of data-aware tags emitted by
#                    make_fixtures.R after seeing the actual data
fixture_ok_ids <- character(0)
enriched_map   <- list()
if (file.exists(MANIFEST)) {
  man <- jsonlite::fromJSON(MANIFEST, simplifyVector = FALSE)
  ok_entries <- Filter(function(e) identical(e$status, "ok"), man$entries)
  fixture_ok_ids <- vapply(ok_entries, function(e) e$id, character(1))
  for (e in ok_entries) {
    et <- unlist(e$enriched_tags)
    if (is.null(et)) et <- character(0)
    enriched_map[[e$id]] <- et
  }
}
cat(sprintf("loaded %d cases (%d executable, %d with fixture)\n",
            length(cases), length(exec_cases), length(fixture_ok_ids)))

# -----------------------------------------------------------------------------
# (a) per-axis feature coverage
# -----------------------------------------------------------------------------
fixture_ok_set <- fixture_ok_ids
fx_cases <- Filter(function(c) c$id %in% fixture_ok_set, exec_cases)

# For fixture-backed coverage, union the case's base (formula-derived) tags
# with any enriched (data-derived) tags emitted at fixture-build time.
case_fx_tags <- function(c) {
  base <- unlist(c$features)
  if (is.null(base)) base <- character(0)
  extra <- enriched_map[[c$id]]
  if (is.null(extra)) extra <- character(0)
  unique(c(base, extra))
}

all_tags    <- unlist(lapply(exec_cases, function(c) unlist(c$features)))
fx_tags     <- unlist(lapply(fx_cases,   case_fx_tags))
tag_counts  <- table(all_tags)
fx_counts   <- table(fx_tags)

axis_rows <- list()
for (section in c("shared", "wr", "lme4", "mgcv")) {
  for (axis in fm[[section]]) {
    for (val in axis$values) {
      id <- sprintf("%s.%s.%s", section, axis$axis, val$id)
      n  <- as.integer(tag_counts[id]);  if (is.na(n))  n  <- 0L
      nf <- as.integer(fx_counts[id]);   if (is.na(nf)) nf <- 0L
      axis_rows[[length(axis_rows) + 1L]] <- list(
        section = section, axis = axis$axis, id = id, n = n, n_fx = nf
      )
    }
  }
}

# -----------------------------------------------------------------------------
# (b) forced_pairs
# -----------------------------------------------------------------------------
pairs <- fm$forced_pairs
pair_hits <- integer(length(pairs))
pair_fx_hits <- integer(length(pairs))
pair_examples <- character(length(pairs))
for (i in seq_along(pairs)) {
  a <- pairs[[i]][[1]]; b <- pairs[[i]][[2]]
  hit_ids <- character(0); fx_hit_ids <- character(0)
  for (c in exec_cases) {
    fs <- unlist(c$features)
    # Pair satisfied (tagged) if both in base tags
    if (a %in% fs && b %in% fs) hit_ids <- c(hit_ids, c$id)
    # Pair satisfied (fixture-backed) if both in base ∪ enriched
    if (c$id %in% fixture_ok_set) {
      merged <- case_fx_tags(c)
      if (a %in% merged && b %in% merged) fx_hit_ids <- c(fx_hit_ids, c$id)
    }
  }
  pair_hits[i] <- length(hit_ids)
  pair_fx_hits[i] <- length(fx_hit_ids)
  pair_examples[i] <- if (length(fx_hit_ids) > 0) fx_hit_ids[1]
                      else if (length(hit_ids) > 0) hit_ids[1]
                      else ""
}

# -----------------------------------------------------------------------------
# (c) dataset assignment
# -----------------------------------------------------------------------------
flat_canon <- unlist(unname(fm$meta$canonical_datasets),
                     recursive = FALSE, use.names = FALSE)
canonical_pkgs <- unique(vapply(flat_canon, `[[`, character(1), "pkg"))
have_data <- list()
for (pkg in list.dirs(DATA, recursive = FALSE, full.names = FALSE)) {
  csvs <- sub("\\.csv$", "", basename(Sys.glob(file.path(DATA, pkg, "*.csv"))))
  have_data[[pkg]] <- csvs
}

resolve_dataset <- function(c) {
  pkg <- c$source_pkg
  rd  <- c$source_rd
  if (!is.null(pkg) && pkg %in% names(have_data)) {
    if (rd %in% have_data[[pkg]]) return(list(pkg = pkg, name = rd))
  }
  # search all pkgs — e.g. 'sleepstudy' lives under 'lme4', 'iris' under 'datasets'
  for (p in names(have_data)) {
    if (rd %in% have_data[[p]]) return(list(pkg = p, name = rd))
  }
  NULL
}

n_resolved <- 0
need_dataset <- list()
for (c in exec_cases) {
  r <- resolve_dataset(c)
  if (!is.null(r)) n_resolved <- n_resolved + 1L
  else need_dataset[[length(need_dataset) + 1L]] <- c
}

# -----------------------------------------------------------------------------
# Write report
# -----------------------------------------------------------------------------
f <- file(OUT, "w")
on.exit(close(f))
w <- function(...) cat(..., "\n", sep = "", file = f)

w("=============================================================================")
w("  COVERAGE AUDIT  —  ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"))
w("=============================================================================")
w("")
w(sprintf("cases total:        %d", length(cases)))
w(sprintf("  executable:       %d", length(exec_cases)))
w(sprintf("  unexecutable:     %d  (covariance-class / parametric / deferred)",
          length(cases) - length(exec_cases)))
w("")
w(sprintf("feature values defined:  %d", length(axis_rows)))
w(sprintf("  tagged (any case):     %d",
          sum(vapply(axis_rows, function(r) r$n > 0, logical(1)))))
w(sprintf("  fixture-backed:        %d",
          sum(vapply(axis_rows, function(r) r$n_fx > 0, logical(1)))))
w(sprintf("forced pairs:            %d", length(pairs)))
w(sprintf("  tagged:                %d", sum(pair_hits > 0)))
w(sprintf("  fixture-backed:        %d", sum(pair_fx_hits > 0)))
w(sprintf("fixtures:                %d / %d executable cases",
          length(fixture_ok_set), length(exec_cases)))
w("")

# ---- (a) axis coverage table ------------------------------------------------
w("-----------------------------------------------------------------------------")
w("  (a) PER-AXIS COVERAGE")
w("-----------------------------------------------------------------------------")
cur_sec <- ""; cur_ax <- ""
for (r in axis_rows) {
  if (r$section != cur_sec) {
    cur_sec <- r$section
    w("")
    w("### [", cur_sec, "]")
  }
  if (r$axis != cur_ax) {
    cur_ax <- r$axis
    vals <- Filter(function(x) x$section == r$section && x$axis == r$axis, axis_rows)
    covered_fx  <- sum(vapply(vals, function(x) x$n_fx > 0, logical(1)))
    covered_tag <- sum(vapply(vals, function(x) x$n > 0,    logical(1)))
    w(sprintf("  axis: %-24s  fx=%d/%d  tagged=%d/%d",
              r$axis, covered_fx, length(vals), covered_tag, length(vals)))
  }
  marker <- if (r$n_fx == 0 && r$n == 0) "  [MISSING]"
            else if (r$n_fx == 0)         "  [tagged but no fixture]"
            else                          ""
  w(sprintf("      %-50s  n=%-3d  fx=%-3d%s", r$id, r$n, r$n_fx, marker))
}

# ---- (b) forced-pair coverage ----------------------------------------------
w("")
w("-----------------------------------------------------------------------------")
w("  (b) FORCED PAIRS")
w("-----------------------------------------------------------------------------")
for (i in seq_along(pairs)) {
  pr <- pairs[[i]]
  status <- if (pair_hits[i] > 0) sprintf("OK  (n=%d, ex=%s)",
                                          pair_hits[i], pair_examples[i])
            else                   "MISSING"
  w(sprintf("  %s  +  %s   %s", pr[[1]], pr[[2]], status))
}

# ---- (c) datasets ----------------------------------------------------------
w("")
w("-----------------------------------------------------------------------------")
w("  (c) DATASET RESOLUTION")
w("-----------------------------------------------------------------------------")
w(sprintf("resolved:    %d / %d executable", n_resolved, length(exec_cases)))
w(sprintf("unresolved:  %d", length(need_dataset)))
w("")
if (length(need_dataset) > 0) {
  w("cases needing a dataset assignment (source_rd has no matching csv):")
  # group by (pkg, source_rd) for brevity
  keys <- vapply(need_dataset, function(c)
    sprintf("%s::%s", c$source_pkg, c$source_rd), character(1))
  tab <- sort(table(keys), decreasing = TRUE)
  for (k in names(tab)) {
    w(sprintf("  %-40s  n=%d", k, tab[[k]]))
  }
}

w("")
w("-----------------------------------------------------------------------------")
w("  (d) AVAILABLE DATASETS (datasets/<pkg>/<name>.csv)")
w("-----------------------------------------------------------------------------")
for (p in sort(names(have_data))) {
  w(sprintf("  %s  (%d):", p, length(have_data[[p]])))
  for (n in sort(have_data[[p]])) w("    ", n)
}

cat(sprintf("\nwrote %s\n", OUT))
