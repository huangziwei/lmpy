# R-side formula benchmark.
#
# Reads benchmarks/results/suite.json produced by bench.py and times the same
# (formula, dataset) pairs against base R, lme4, and mgcv.
#
# What each library does here:
#   - base        : stats::model.matrix(terms(formula), data)   [parametric X]
#   - lme4        : lme4::lFormula(formula, data)               [X + Zt setup]
#   - mgcv        : mgcv::gam(formula, data, fit=FALSE)         [X + smooth bases]
#
# Libraries that don't semantically handle a given fixture class are recorded
# as status="unsupported". Timing uses replicate() + Sys.time() rather than
# microbenchmark (not installed); we take min/median/max of `reps` runs.

suppressPackageStartupMessages({
  library(jsonlite)
})

script_dir <- local({
  # source()'d: sys.frame(1)$ofile; Rscript: --file=...; fallback: getwd().
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(normalizePath(dirname(ofile)))
  cmd <- commandArgs(trailingOnly = FALSE)
  hit <- grep("^--file=", cmd, value = TRUE)
  if (length(hit) > 0) return(normalizePath(dirname(sub("^--file=", "", hit[[1]]))))
  getwd()
})
root      <- normalizePath(file.path(script_dir, ".."))
suite_js  <- file.path(root, "benchmarks", "results", "suite.json")
out_path  <- file.path(root, "benchmarks", "results", "r.csv")
data_root <- file.path(root, "datasets")

args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(key, default = NULL) {
  hit <- grep(paste0("^--", key, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^--", key, "="), "", hit[[1]])
}
reps    <- as.integer(parse_arg("reps", "7"))
warmup  <- as.integer(parse_arg("warmup", "1"))
scales  <- as.integer(strsplit(parse_arg("scales", "1"), ",")[[1]])
libs_in <- parse_arg("libs", NULL)
kinds_in<- parse_arg("kinds", NULL)
limit   <- suppressWarnings(as.integer(parse_arg("limit", NA)))

if (!file.exists(suite_js)) stop("run bench.py first (suite.json missing)")
suite <- fromJSON(suite_js, simplifyDataFrame = FALSE)

if (!is.null(kinds_in)) {
  keep <- strsplit(kinds_in, ",")[[1]]
  suite <- Filter(function(s) s$kind %in% keep, suite)
}
if (!is.na(limit) && length(suite) > limit) suite <- suite[seq_len(limit)]

versions <- list(
  base = paste(R.version$major, R.version$minor, sep="."),
  lme4 = tryCatch(as.character(packageVersion("lme4")), error=function(e) NA),
  mgcv = tryCatch(as.character(packageVersion("mgcv")), error=function(e) NA)
)

df_cache <- new.env(hash = TRUE, parent = emptyenv())

# Parse one CSV row, returning a list of (text, was_quoted) per field. We need
# this because read.csv discards the quoting signal and will coerce "1","2","3"
# to integer — even though R's write.csv only quotes character/factor columns.
.split_csv_row <- function(line) {
  chars <- strsplit(line, "", fixed = TRUE)[[1]]
  fields <- list(); buf <- character(0)
  in_quotes <- FALSE; field_quoted <- FALSE
  n <- length(chars); i <- 1L
  while (i <= n) {
    ch <- chars[[i]]
    if (in_quotes) {
      if (ch == "\"") {
        if (i + 1L <= n && chars[[i + 1L]] == "\"") {
          buf <- c(buf, "\""); i <- i + 2L; next
        }
        in_quotes <- FALSE
      } else {
        buf <- c(buf, ch)
      }
    } else {
      if (ch == "\"") {
        in_quotes <- TRUE; field_quoted <- TRUE
      } else if (ch == ",") {
        fields[[length(fields) + 1L]] <- list(
          text = paste(buf, collapse = ""), quoted = field_quoted
        )
        buf <- character(0); field_quoted <- FALSE
      } else {
        buf <- c(buf, ch)
      }
    }
    i <- i + 1L
  }
  fields[[length(fields) + 1L]] <- list(
    text = paste(buf, collapse = ""), quoted = field_quoted
  )
  fields
}

.detect_quoted_cols <- function(path, max_rows = 50) {
  lines <- readLines(path, n = max_rows + 1L, warn = FALSE)
  if (length(lines) < 2) return(character())
  header <- vapply(.split_csv_row(lines[[1]]), function(x) x$text, character(1))
  quoted <- logical(length(header))
  for (line in lines[-1]) {
    if (nchar(line) == 0) next
    fields <- .split_csv_row(line)
    for (j in seq_len(min(length(fields), length(quoted)))) {
      if (isTRUE(fields[[j]]$quoted)) quoted[[j]] <- TRUE
    }
  }
  # Drop empty-named columns (R's write.csv row.names column). read.csv
  # synthesises "X" or treats it as row.names depending on options; either
  # way, it's never referenced by formulas.
  cols <- header[quoted]
  cols[nzchar(cols)]
}

load_df <- function(ds) {
  if (!exists(ds, envir = df_cache, inherits = FALSE)) {
    parts <- strsplit(ds, "/", fixed = TRUE)[[1]]
    path  <- file.path(data_root, parts[[1]], paste0(parts[[2]], ".csv"))
    quoted_cols <- .detect_quoted_cols(path)
    col_classes <- setNames(rep("character", length(quoted_cols)), quoted_cols)
    # stringsAsFactors=TRUE restores factor semantics; colClasses forces the
    # quoted columns through the character -> factor path even if they're all
    # digits like "1","2","3","4".
    d <- if (length(col_classes)) {
      read.csv(path, stringsAsFactors = TRUE, colClasses = col_classes)
    } else {
      read.csv(path, stringsAsFactors = TRUE)
    }
    # read.csv with colClasses="character" doesn't auto-factor; do it ourselves.
    for (col in quoted_cols) d[[col]] <- factor(d[[col]])
    assign(ds, d, envir = df_cache)
  }
  get(ds, envir = df_cache, inherits = FALSE)
}
scale_df <- function(df, k) if (k <= 1) df else df[rep(seq_len(nrow(df)), k), , drop=FALSE]

time_fn <- function(f, reps, warmup) {
  for (i in seq_len(warmup)) f()
  times <- replicate(reps, {
    t0 <- Sys.time()
    f()
    as.numeric(difftime(Sys.time(), t0, units = "secs"))
  })
  times <- sort(times)
  list(min_s = times[[1]],
       median_s = times[[ceiling(length(times)/2)]],
       max_s = times[[length(times)]])
}

# --- per-library callables --------------------------------------------------

run_base <- function(formula_str, data, kind, scope = "full") {
  f  <- stats::as.formula(formula_str)
  # `base` means model.matrix on the parametric RHS only. For mgcv formulas,
  # strip s()/te()/ti()/t2() via mgcv::interpret.gam so model.matrix doesn't
  # try to evaluate smooth constructors.
  if (kind == "mgcv") {
    pf <- mgcv::interpret.gam(f)$pf
    tt <- stats::terms(pf, data = data)
  } else {
    tt <- stats::terms(f, data = data)
  }
  stats::model.matrix(tt, data = data)
}

run_lme4 <- function(formula_str, data, kind, scope = "full") {
  f <- stats::as.formula(formula_str)
  ctl <- lme4::lmerControl(
    check.rankX        = "silent.drop.cols",
    check.nobs.vs.nlev = "ignore",
    check.nobs.vs.nRE  = "ignore",
    check.nobs.vs.rankZ= "ignore"
  )
  if (identical(scope, "parametric")) {
    # Strip the (...|group) bars so terms() sees just the parametric RHS.
    nb <- lme4::nobars(f)
    if (is.null(nb)) nb <- stats::as.formula(paste(deparse(f[[2]]), "~ 1"))
    stats::model.matrix(stats::terms(nb, data = data), data = data)
  } else {
    lme4::lFormula(f, data = data, control = ctl)
  }
}

run_mgcv <- function(formula_str, data, kind, scope = "full") {
  f <- stats::as.formula(formula_str)
  if (identical(scope, "parametric")) {
    pf <- mgcv::interpret.gam(f)$pf
    stats::model.matrix(stats::terms(pf, data = data), data = data)
  } else {
    mgcv::gam(f, data = data, fit = FALSE)
  }
}

lib_defs <- list(
  list(name = "base",  fn = run_base,  kinds = c("wr", "mgcv")),
  list(name = "lme4",  fn = run_lme4,  kinds = c("lme4")),
  list(name = "mgcv",  fn = run_mgcv,  kinds = c("mgcv"))
)

.is_data_insufficient <- function(msg) {
  pats <- c("fewer unique", "insufficient unique",
            "doesn't match margin count",
            "insufficient data", "too few")
  any(vapply(pats, function(p) grepl(p, msg, fixed = TRUE), logical(1)))
}
if (!is.null(libs_in)) {
  keep <- strsplit(libs_in, ",")[[1]]
  lib_defs <- Filter(function(l) l$name %in% keep, lib_defs)
}

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
con <- file(out_path, open = "w")
write.table(
  data.frame(library=character(), version=character(), fixture_id=character(),
             kind=character(), formula=character(), dataset=character(),
             n_rows=integer(), scale=integer(), scope=character(),
             status=character(), error=character(), n_reps=integer(),
             min_s=numeric(), median_s=numeric(), max_s=numeric()),
  con, sep = ",", row.names = FALSE, qmethod = "double"
)

total <- length(suite) * length(lib_defs) * length(scales)
i <- 0L
t0 <- Sys.time()

write_row <- function(row) {
  write.table(
    as.data.frame(row, stringsAsFactors = FALSE),
    con, sep = ",", row.names = FALSE, col.names = FALSE,
    qmethod = "double", append = TRUE
  )
  flush(con)
}

for (fx in suite) {
  df_base <- tryCatch(load_df(fx$dataset), error = function(e) NULL)
  if (is.null(df_base)) {
    for (s in scales) for (ld in lib_defs) {
      i <- i + 1L
      write_row(list(
        library=ld$name, version=as.character(if (is.null(versions[[ld$name]])) "" else versions[[ld$name]]),
        fixture_id=fx$id, kind=fx$kind, formula=fx$formula, dataset=fx$dataset,
        n_rows=0L, scale=s, status="error", error="dataset not found",
        n_reps=0L, min_s=NA_real_, median_s=NA_real_, max_s=NA_real_
      ))
    }
    next
  }
  for (s in scales) {
    df <- scale_df(df_base, s)
    for (ld in lib_defs) {
      i <- i + 1L
      status <- "ok"; err <- ""; scope <- "full"
      t <- list(min_s=NA_real_, median_s=NA_real_, max_s=NA_real_)
      if (!(fx$kind %in% ld$kinds)) {
        status <- "unsupported"
        err <- sprintf("kind=%s not in lib=%s scope", fx$kind, ld$name)
      } else {
        attempt <- function(scope_arg) tryCatch(
          list(ok = TRUE, t = time_fn(function() ld$fn(fx$formula, df, fx$kind, scope_arg), reps, warmup)),
          error = function(e) list(ok = FALSE, msg = conditionMessage(e))
        )
        res <- attempt("full")
        if (isTRUE(res$ok)) {
          t <- res$t
        } else if (.is_data_insufficient(res$msg)) {
          # Fall back to parametric: matches what the fixture oracle captured.
          res2 <- attempt("parametric")
          if (isTRUE(res2$ok)) {
            t <- res2$t; scope <- "parametric"
            err <- paste0("degraded: ", substr(res$msg, 1, 200))
          } else {
            status <- "error"; err <- substr(res2$msg, 1, 300)
          }
        } else {
          msg <- tolower(res$msg)
          status <- if (any(vapply(
            c("not supported","not implemented","unsupported","unknown",
              "no applicable","can't","cannot",
              "could not find function"),
            function(k) grepl(k, msg, fixed = TRUE), logical(1)
          ))) "unsupported" else "error"
          err <- substr(res$msg, 1, 300)
        }
      }
      row <- list(
        library=ld$name,
        version=as.character(if (is.null(versions[[ld$name]])) "" else versions[[ld$name]]),
        fixture_id=fx$id, kind=fx$kind, formula=fx$formula, dataset=fx$dataset,
        n_rows=nrow(df), scale=s, scope=scope, status=status, error=err,
        n_reps=if (status=="ok") reps else 0L,
        min_s=t$min_s, median_s=t$median_s, max_s=t$max_s
      )
      write_row(row)
      elapsed <- as.numeric(difftime(Sys.time(), t0, units="secs"))
      t_str <- if (is.na(t$min_s)) "-" else sprintf("%.2e", t$min_s)
      cat(sprintf("[%4d/%d] %-10s %-5s %-10s s=%-3d n=%-6d %-11s %s  [%5.1fs]\n",
                  i, total, ld$name, fx$kind, fx$id, s, nrow(df),
                  status, t_str, elapsed))
    }
  }
}

close(con)
cat(sprintf("\nWrote %s\n", out_path))
cat(sprintf("Versions: %s\n", toJSON(versions, auto_unbox=TRUE)))
