#!/usr/bin/env Rscript
# Export canonical datasets referenced by corpus/feature_matrix.yaml
# meta.canonical_datasets to datasets/<pkg>/<name>.csv.
#
# Skips missing packages gracefully and prints an install hint.
# Special-cases a few entries (e.g. mgcv gamSim synthetic, lme4 InstEval subsample).

suppressPackageStartupMessages({
  library(yaml)
})

OUT_ROOT <- "datasets"
YAML_PATH <- "corpus/feature_matrix.yaml"

fm <- yaml::read_yaml(YAML_PATH)
entries <- do.call(c, unname(fm$meta$canonical_datasets))

write_csv <- function(df, pkg, name) {
  dir.create(file.path(OUT_ROOT, pkg), showWarnings = FALSE, recursive = TRUE)
  path <- file.path(OUT_ROOT, pkg, paste0(name, ".csv"))
  write.csv(df, path, row.names = FALSE, na = "NA")
  cat(sprintf("  ok    %-24s  %-20s  n=%-5d  p=%d\n", pkg, name,
              nrow(df), ncol(df)))
}

# ---- special fetchers -------------------------------------------------------
# Some "datasets" need custom loading (not just data()).
special <- list(
  # mgcv::gamSim(1) returns synthetic data used throughout mgcv examples
  list(pkg = "mgcv", name = "gamSim_eg1",
       fn = function() { set.seed(2); mgcv::gamSim(eg = 1, n = 400, verbose = FALSE) }),
  # InstEval is huge (~73k rows); also export a stratified 5k-row subsample
  list(pkg = "lme4", name = "InstEval_sample",
       fn = function() {
         set.seed(3)
         d <- lme4::InstEval
         d[sample(nrow(d), 5000), ]
       })
)

# ---- main loop --------------------------------------------------------------
missing_pkgs <- character(0)
skipped      <- 0
ok           <- 0

cat("Exporting canonical datasets\n")
for (e in entries) {
  pkg <- e$pkg; name <- e$name

  # special handlers
  sp_match <- Filter(function(s) s$pkg == pkg && s$name == name, special)
  if (length(sp_match) > 0) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(sprintf("  skip  %-24s  %-20s  (package not installed)\n", pkg, name))
      missing_pkgs <- c(missing_pkgs, pkg)
      skipped <- skipped + 1
      next
    }
    df <- tryCatch(sp_match[[1]]$fn(), error = function(err) err)
    if (inherits(df, "error")) {
      cat(sprintf("  ERR   %-24s  %-20s  %s\n", pkg, name, conditionMessage(df)))
      skipped <- skipped + 1
      next
    }
    write_csv(df, pkg, name)
    ok <- ok + 1
    next
  }

  # default: data() loader
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  skip  %-24s  %-20s  (package not installed)\n", pkg, name))
    missing_pkgs <- c(missing_pkgs, pkg)
    skipped <- skipped + 1
    next
  }

  df <- tryCatch({
    e_env <- new.env()
    data(list = name, package = pkg, envir = e_env)
    obj <- get(name, envir = e_env)
    if (is.data.frame(obj)) obj else as.data.frame(obj)
  }, error = function(err) err, warning = function(w) w)

  if (inherits(df, "error") || inherits(df, "warning")) {
    cat(sprintf("  ERR   %-24s  %-20s  %s\n", pkg, name,
                conditionMessage(df)))
    skipped <- skipped + 1
    next
  }
  write_csv(df, pkg, name)
  ok <- ok + 1
}

# also export the InstEval subsample (not listed in the YAML, but useful)
if (requireNamespace("lme4", quietly = TRUE)) {
  sp <- special[[2]]
  df <- sp$fn()
  write_csv(df, sp$pkg, sp$name)
  ok <- ok + 1
}

cat(sprintf("\nexported %d, skipped %d\n", ok, skipped))

missing_pkgs <- unique(missing_pkgs)
if (length(missing_pkgs) > 0) {
  cat("\nTo include skipped datasets, install:\n")
  cat(sprintf("  Rscript -e 'install.packages(c(%s), repos=\"https://cloud.r-project.org\")'\n",
              paste(sprintf("\"%s\"", missing_pkgs), collapse = ", ")))
}
