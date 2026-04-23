#!/usr/bin/env Rscript
# Scrape formulas from installed R package help files (Rd examples).
#
# Strategy: for each package, pull all Rd help pages via tools::Rd_db, render
# the \examples section to runnable R code via tools::Rd2ex, parse that code
# to an AST, and walk the AST for calls to model-fitting functions. The first
# positional arg (or the named `formula` arg) is the formula; deparse it.
#
# Output: corpus/harvested_raw.yaml — raw, de-duplicated, grouped by source
# package. Intended for manual review and tagging with feature:value ids from
# corpus/feature_matrix.yaml. Not consumed directly by the fixture generator.

suppressPackageStartupMessages({
  library(tools)
  library(yaml)
})

OUT <- "corpus/harvested_raw.yaml"
dir.create(dirname(OUT), showWarnings = FALSE, recursive = TRUE)

# Which packages to scrape, and which function-calls within them carry a
# formula as first arg (or as arg named `formula`, `fixed`, or `form`).
TARGETS <- list(
  stats = c("lm", "glm", "aov", "model.matrix", "model.frame"),
  lme4  = c("lmer", "glmer", "nlmer", "lFormula", "glFormula", "mkReTrms"),
  mgcv  = c("gam", "bam", "gamm", "s", "te", "ti", "t2"),
  nlme  = c("lme", "gls", "nlme"),
  MASS  = c("lm", "glm"),
  # book-companion packages: rich lm/gam example pool
  faraway = c("lm", "glm", "aov"),
  gamair  = c("gam", "bam", "gamm", "lm"),
  splines = c()          # kept so Rd_db is probed for ?bs / ?ns examples
)

# named-arg fallback keys (in order of precedence)
FORMULA_ARGS <- c("formula", "fixed", "form")

# ---- helpers ----------------------------------------------------------------
rd_to_example_code <- function(rd) {
  tmpfile <- tempfile(fileext = ".R")
  res <- tryCatch(
    tools::Rd2ex(rd, out = tmpfile, defines = NULL),
    error = function(e) NULL,
    warning = function(w) NULL
  )
  if (is.null(res) || !file.exists(tmpfile)) return(character(0))
  code <- tryCatch(readLines(tmpfile, warn = FALSE),
                   error = function(e) character(0))
  unlink(tmpfile)
  code
}

is_formula_expr <- function(e) {
  is.call(e) && is.symbol(e[[1]]) && as.character(e[[1]]) == "~"
}

safe_elt <- function(lst, i) {
  # Access list element; return NULL for missing-arg symbols or on error.
  x <- tryCatch(lst[[i]], error = function(e) NULL)
  if (is.null(x)) return(NULL)
  if (is.symbol(x) && !nzchar(as.character(x))) return(NULL)
  x
}

# Given a parsed call, return the formula expression if present, else NULL.
extract_formula_arg <- function(call_expr) {
  args <- as.list(call_expr)[-1]
  if (length(args) == 0) return(NULL)
  nm <- names(args); if (is.null(nm)) nm <- rep("", length(args))

  for (key in FORMULA_ARGS) {
    idx <- which(nm == key)
    if (length(idx) > 0) {
      cand <- safe_elt(args, idx[1])
      if (!is.null(cand) && is_formula_expr(cand)) return(cand)
    }
  }
  for (i in which(nm == "")) {
    cand <- safe_elt(args, i)
    if (is.null(cand)) next
    if (is_formula_expr(cand)) return(cand)
    break
  }
  NULL
}

is_empty_sym <- function(x) {
  is.symbol(x) && !nzchar(as.character(x))
}

walk_for_formulas <- function(node, target_fns, found) {
  tryCatch({
    if (missing(node) || is_empty_sym(node)) return(found)

    if (is.call(node)) {
      head <- node[[1]]
      fname <- NULL
      if (is.symbol(head)) {
        fname <- as.character(head)
      } else if (is.call(head) && length(head) == 3 &&
                 is.symbol(head[[1]]) &&
                 as.character(head[[1]]) %in% c("::", ":::") &&
                 is.symbol(head[[3]])) {
        fname <- as.character(head[[3]])
      }
      if (!is.null(fname) && fname %in% target_fns) {
        f <- extract_formula_arg(node)
        if (!is.null(f)) {
          found[[length(found) + 1L]] <- list(fn = fname, formula = f)
        }
      }
    }

    if (is.call(node) || is.pairlist(node)) {
      for (i in seq_along(node)) {
        ch <- safe_elt(node, i)
        if (is.null(ch)) next
        found <- walk_for_formulas(ch, target_fns, found)
      }
    }
    found
  }, error = function(e) found)
}

scrape_package <- function(pkg, target_fns) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  skip  %-10s  (not installed)\n", pkg))
    return(list())
  }
  db <- tryCatch(tools::Rd_db(pkg), error = function(e) NULL)
  if (is.null(db)) {
    cat(sprintf("  skip  %-10s  (no Rd_db)\n", pkg))
    return(list())
  }
  out <- list()
  for (rd_name in names(db)) {
    code <- rd_to_example_code(db[[rd_name]])
    if (length(code) == 0) next
    # parse ignoring any errors — Rd examples sometimes contain dontrun/partial
    exprs <- tryCatch(parse(text = code, keep.source = FALSE),
                      error = function(e) NULL)
    if (is.null(exprs)) next
    found <- list()
    for (e in as.list(exprs)) {
      found <- walk_for_formulas(e, target_fns, found)
    }
    for (f in found) {
      out[[length(out) + 1L]] <- list(
        pkg = pkg,
        fn  = f$fn,
        formula = paste(deparse(f$formula, width.cutoff = 500L), collapse = " "),
        source_rd = sub("\\.Rd$", "", rd_name)
      )
    }
  }
  cat(sprintf("  ok    %-10s  %d formula(s) from %d help files\n",
              pkg, length(out), length(db)))
  out
}

# ---- run --------------------------------------------------------------------
cat("Harvesting formulas from installed package help files\n")
all_found <- list()
for (pkg in names(TARGETS)) {
  found <- scrape_package(pkg, TARGETS[[pkg]])
  all_found <- c(all_found, found)
}

# Normalize whitespace and drop exact duplicates (same formula string).
for (i in seq_along(all_found)) {
  f <- all_found[[i]]$formula
  f <- gsub("\\s+", " ", f)
  f <- trimws(f)
  all_found[[i]]$formula <- f
}
keys <- vapply(all_found, function(x) paste(x$pkg, x$fn, x$formula, sep = "|"),
               character(1))
all_found <- all_found[!duplicated(keys)]

# Group for nicer YAML output.
by_pkg <- split(all_found, vapply(all_found, `[[`, character(1), "pkg"))
for (pkg in names(by_pkg)) {
  by_pkg[[pkg]] <- lapply(by_pkg[[pkg]], function(e) {
    list(fn = e$fn, formula = e$formula, source_rd = e$source_rd,
         features = list(), notes = "")
  })
}

yaml::write_yaml(
  list(
    meta = list(
      generated_by = "scripts/harvest.R",
      generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
      r_version = R.version.string,
      packages = lapply(names(TARGETS), function(p) {
        if (requireNamespace(p, quietly = TRUE))
          list(pkg = p, version = as.character(packageVersion(p)))
        else
          list(pkg = p, version = "NOT_INSTALLED")
      }),
      count_total = length(all_found),
      counts_by_pkg = lapply(by_pkg, length)
    ),
    formulas = by_pkg
  ),
  file = OUT
)
cat(sprintf("\nwrote %d unique formulas to %s\n", length(all_found), OUT))
