#!/usr/bin/env Rscript
# Export every usable data.frame from a list of R packages into
# datasets/<pkg>/<name>.csv, plus a couple of special fetchers that need
# custom construction (mgcv::gamSim synthetic, lme4::InstEval subsample).
#
# Idempotent: skips any file that already exists, so re-running is cheap
# and won't clobber manual edits. Synthetic seed_*.csv files come from
# tests/scripts/synthetic_data.R and live under datasets/synthetic/.

suppressPackageStartupMessages({
  library(tools)
  library(jsonlite)
})

OUT_ROOT <- "datasets"
# Packages we still bundle as CSVs because rdatasets doesn't carry them.
# MASS, nlme, datasets, and most of lme4 are sourced from rdatasets at runtime
# (see hea/data.py and the cleanup in /Users/ziweih/Works/hea/datasets/),
# so re-exporting them here would just re-fill slots we just deleted.
PKGS <- c("mgcv", "faraway", "gamair", "splines", "stats")
# A few lme4 datasets aren't carried by rdatasets — keep these bundled.
LME4_KEEP <- c("InstEval_sample", "cbpp2", "culcitalogreg", "culcitalvolume",
               "ergoStool", "gopherdat2", "salamander", "schizophrenia", "toenail")

dir.create(OUT_ROOT, showWarnings = FALSE, recursive = TRUE)

# ---- special fetchers -------------------------------------------------------
# Dataset-like things that need a custom constructor, not just data().
special <- list(
  list(pkg = "mgcv", name = "gamSim_eg1",
       fn = function() { set.seed(2); mgcv::gamSim(eg = 1, n = 400, verbose = FALSE) }),
  list(pkg = "mgcv", name = "gamSim_eg2",
       fn = function() { set.seed(2); mgcv::gamSim(eg = 2, n = 400, verbose = FALSE)$data }),
  list(pkg = "mgcv", name = "gamSim_eg4",
       fn = function() { set.seed(2); mgcv::gamSim(eg = 4, n = 400, verbose = FALSE) }),
  list(pkg = "mgcv", name = "gamSim_eg5",
       fn = function() { set.seed(2); mgcv::gamSim(eg = 5, n = 400, verbose = FALSE) }),
  list(pkg = "mgcv", name = "gamSim_eg6",
       fn = function() { set.seed(2); mgcv::gamSim(eg = 6, n = 400, verbose = FALSE) }),
  list(pkg = "lme4", name = "InstEval_sample",
       fn = function() {
         set.seed(3); d <- lme4::InstEval; d[sample(nrow(d), 5000), ]
       })
)

# Map pkg name -> output subdir. R's built-in `datasets::` package goes under
# `datasets/R/` to avoid `datasets/datasets/` path duplication.
pkg_dir <- function(pkg) if (pkg == "datasets") "R" else pkg

# True when row.names are just sequential "1".."n" — i.e. R's default and
# carry no actual information. Anything else (character names like the
# Galápagos islands in faraway::gala, or non-sequential integer IDs from
# subsetted frames in gamair) is a row identifier worth preserving.
trivial_rownames <- function(df) {
  rn <- rownames(df)
  is.null(rn) || identical(rn, as.character(seq_len(nrow(df))))
}

write_csv <- function(df, pkg, name) {
  sub <- pkg_dir(pkg)
  dir.create(file.path(OUT_ROOT, sub), showWarnings = FALSE, recursive = TRUE)
  path <- file.path(OUT_ROOT, sub, paste0(name, ".csv"))
  if (file.exists(path)) return(FALSE)
  if (!trivial_rownames(df)) {
    df <- cbind(rowname = rownames(df), df, stringsAsFactors = FALSE)
  }
  write.csv(df, path, row.names = FALSE, na = "NA")
  write_schema(df, pkg, name)
  TRUE
}

# Sidecar <name>.schema.json next to the CSV. CSV loses factor type, so we
# record which columns to re-`factor()` on read (R side) or re-cast to
# pl.Enum (Python side). No-factor frames get no schema file.
write_schema <- function(df, pkg, name) {
  sub <- pkg_dir(pkg)
  fac_cols <- names(df)[vapply(df, is.factor, logical(1))]
  path <- file.path(OUT_ROOT, sub, paste0(name, ".schema.json"))
  if (length(fac_cols) == 0L) {
    if (file.exists(path)) file.remove(path)
    return(invisible(FALSE))
  }
  facs <- list()
  for (col in fac_cols) {
    facs[[col]] <- list(
      levels  = as.character(levels(df[[col]])),
      ordered = is.ordered(df[[col]])
    )
  }
  writeLines(
    jsonlite::toJSON(list(factors = facs), auto_unbox = TRUE, pretty = TRUE),
    path
  )
  invisible(TRUE)
}

# ---- main -------------------------------------------------------------------
ok <- 0; skipped <- 0; non_df <- 0; err <- 0

# 1. Bulk export via data() for each package
for (pkg in PKGS) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  skip %s (not installed)\n", pkg)); next
  }
  idx <- tryCatch(data(package = pkg)$results, error = function(e) NULL)
  if (is.null(idx) || nrow(idx) == 0) next

  for (i in seq_len(nrow(idx))) {
    name <- sub(" \\(.*$", "", idx[i, "Item"])
    path <- file.path(OUT_ROOT, pkg_dir(pkg), paste0(name, ".csv"))
    if (file.exists(path)) { skipped <- skipped + 1; next }

    df <- tryCatch({
      e <- new.env()
      data(list = name, package = pkg, envir = e)
      get(name, envir = e)
    }, error = function(err) err, warning = function(w) w)

    if (inherits(df, "error") || inherits(df, "warning")) {
      err <- err + 1; next
    }
    if (!is.data.frame(df)) {
      if (is.matrix(df)) df <- as.data.frame(df)
      else               { non_df <- non_df + 1; next }
    }
    if (nrow(df) < 3 || ncol(df) < 1) { non_df <- non_df + 1; next }

    if (write_csv(df, pkg, name)) ok <- ok + 1 else skipped <- skipped + 1
  }
}

# 1b. lme4 — only the items rdatasets doesn't carry. Most of lme4 is sourced
# from rdatasets at runtime (see _RDATASETS_PKG_ALIAS in hea/data.py).
if (requireNamespace("lme4", quietly = TRUE)) {
  for (name in LME4_KEEP) {
    path <- file.path(OUT_ROOT, "lme4", paste0(name, ".csv"))
    if (file.exists(path)) { skipped <- skipped + 1; next }
    df <- tryCatch({
      e <- new.env()
      data(list = name, package = "lme4", envir = e)
      get(name, envir = e)
    }, error = function(err) err, warning = function(w) w)
    if (inherits(df, "error") || inherits(df, "warning")) { err <- err + 1; next }
    if (!is.data.frame(df)) { non_df <- non_df + 1; next }
    if (write_csv(df, "lme4", name)) ok <- ok + 1
  }
}

# 2. Special fetchers
for (sp in special) {
  if (!requireNamespace(sp$pkg, quietly = TRUE)) next
  path <- file.path(OUT_ROOT, pkg_dir(sp$pkg), paste0(sp$name, ".csv"))
  if (file.exists(path)) { skipped <- skipped + 1; next }
  df <- tryCatch(sp$fn(), error = function(e) e)
  if (inherits(df, "error")) { err <- err + 1; next }
  if (write_csv(df, sp$pkg, sp$name)) ok <- ok + 1
}

cat(sprintf("\nexported %d new datasets (already had %d; %d non-df/degenerate; %d errors)\n",
            ok, skipped, non_df, err))
