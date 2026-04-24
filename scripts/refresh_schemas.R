#!/usr/bin/env Rscript
# Emit sidecar <name>.schema.json next to every datasets/<pkg>/<name>.csv that
# originates from an R data.frame with factor columns. CSV round-trips lose
# factor type (quoted numeric-ish levels re-parse as integer, character levels
# as character), so downstream code has no way to know which columns R treated
# as factors unless we write it explicitly.
#
# Strategy: for each CSV on disk, re-fetch the source data.frame from R, look
# at is.factor() / is.ordered() per column, and dump {factors: {col: {levels,
# ordered}}}. Datasets that aren't found in `data()` and aren't in the
# `special` list are skipped silently — schemas only exist where needed.
#
# Idempotent: re-running overwrites schemas; no-op for CSVs with no factors.

suppressPackageStartupMessages({
  library(tools)
  library(jsonlite)
})

OUT_ROOT <- "datasets"
PKGS <- c("datasets", "MASS", "nlme", "lme4", "mgcv", "faraway", "gamair",
          "splines", "stats")

# Mirrors scripts/export_data.R: custom-fetched data.frames keyed by (pkg, name).
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

# datasets/R/ mirrors the built-in `datasets` package.
pkg_dir <- function(pkg) if (pkg == "datasets") "R" else pkg

schema_path <- function(pkg, name) {
  file.path(OUT_ROOT, pkg_dir(pkg), paste0(name, ".schema.json"))
}

# Emit (or delete) schema for df at (pkg, name). Returns TRUE if any factors.
emit_schema <- function(df, pkg, name) {
  fac_cols <- names(df)[vapply(df, is.factor, logical(1))]
  path <- schema_path(pkg, name)
  if (length(fac_cols) == 0L) {
    # Keep tree tidy: no schema file means "no factors".
    if (file.exists(path)) file.remove(path)
    return(FALSE)
  }
  facs <- list()
  for (col in fac_cols) {
    facs[[col]] <- list(
      levels  = as.character(levels(df[[col]])),
      ordered = is.ordered(df[[col]])
    )
  }
  writeLines(
    toJSON(list(factors = facs), auto_unbox = TRUE, pretty = TRUE),
    path
  )
  TRUE
}

# ---- backfill ---------------------------------------------------------------

n_emitted <- 0; n_nofac <- 0; n_missing <- 0

# 1. data()-based datasets
for (pkg in PKGS) {
  if (!requireNamespace(pkg, quietly = TRUE)) next
  idx <- tryCatch(data(package = pkg)$results, error = function(e) NULL)
  if (is.null(idx) || nrow(idx) == 0) next

  for (i in seq_len(nrow(idx))) {
    name <- sub(" \\(.*$", "", idx[i, "Item"])
    csv  <- file.path(OUT_ROOT, pkg_dir(pkg), paste0(name, ".csv"))
    if (!file.exists(csv)) next

    df <- tryCatch({
      e <- new.env()
      data(list = name, package = pkg, envir = e)
      get(name, envir = e)
    }, error = function(err) NULL, warning = function(w) NULL)
    if (is.null(df) || !is.data.frame(df)) { n_missing <- n_missing + 1; next }

    if (emit_schema(df, pkg, name)) n_emitted <- n_emitted + 1 else n_nofac <- n_nofac + 1
  }
}

# 2. Special fetchers
for (sp in special) {
  csv <- file.path(OUT_ROOT, pkg_dir(sp$pkg), paste0(sp$name, ".csv"))
  if (!file.exists(csv)) next
  if (!requireNamespace(sp$pkg, quietly = TRUE)) next
  df <- tryCatch(sp$fn(), error = function(e) NULL)
  if (is.null(df) || !is.data.frame(df)) { n_missing <- n_missing + 1; next }
  if (emit_schema(df, sp$pkg, sp$name)) n_emitted <- n_emitted + 1 else n_nofac <- n_nofac + 1
}

cat(sprintf("schemas: %d emitted (factor cols), %d skipped (no factors), %d not-fetchable\n",
            n_emitted, n_nofac, n_missing))
