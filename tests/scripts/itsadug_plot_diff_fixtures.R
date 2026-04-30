#!/usr/bin/env Rscript
# Build oracle fixtures for hea.gam.get_difference / plot_diff against
# itsadug::get_difference. One synthetic dataset, one model fit, several
# (comp, cond, f, sim.ci, rm.ranef) cases — each case dumps:
#
#   data.csv          fitting data (shared across cases per dataset, also dumped here)
#   args.json         comp / cond / f / sim.ci / rm.ranef passed to itsadug
#   diff_table.csv    output of itsadug::get_difference (cond cols + difference + CI [+ sim.CI])
#   beta.csv          coef(model)
#   Vp.mtx            vcov(model)                                  (frequentist)
#   Vc.mtx            vcov(model, freq=FALSE, unconditional=TRUE)  (Vp + sp-correction)
#
# Cases live under tests/fixtures/itsadug_plot_diff/<case_id>/. The Python
# test re-fits the same model in hea, calls m.get_difference with the same
# args, and compares element-wise. Beta/Vp/Vc are snapshotted so we can
# verify the upstream model alignment separately from the difference math.
#
# Re-run idempotently: `Rscript tests/scripts/itsadug_plot_diff_fixtures.R`.

suppressPackageStartupMessages({
  library(jsonlite)
  library(Matrix)
  library(mgcv)
  library(itsadug)
})

OUT_ROOT <- "tests/fixtures/itsadug_plot_diff"
dir.create(OUT_ROOT, showWarnings = FALSE, recursive = TRUE)

write_mm <- function(m, path) {
  Matrix::writeMM(Matrix::Matrix(as.matrix(m), sparse = TRUE), path)
}

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
#  - x        numeric covariate (smooth axis)
#  - group    3-level factor in s(x, by=group); also enters parametrically
#  - cohort   2-level factor entering parametrically — exercises the
#             "hold-at-typical-value" path for non-comp factors
#  - y        gaussian
make_data <- function() {
  set.seed(20260430)
  n_per <- 120
  groups  <- rep(c("A", "B", "C"), each = n_per)
  cohorts <- sample(c("Y", "Z"), length(groups), replace = TRUE)
  x <- runif(length(groups), 0, 10)

  # group-specific smooth signal — A: sin, B: gentle quadratic, C: flat
  fA <- function(x) 1.5 * sin(0.7 * x)
  fB <- function(x) 0.04 * (x - 5)^2 - 0.5
  fC <- function(x) rep(0.2, length(x))
  mu_smooth <- ifelse(groups == "A", fA(x),
              ifelse(groups == "B", fB(x), fC(x)))
  mu_param  <- 0.0 + ifelse(groups == "A", 0.0,
                     ifelse(groups == "B", 0.4, -0.3)) +
               ifelse(cohorts == "Y", 0.0, 0.2)
  y <- mu_smooth + mu_param + rnorm(length(x), sd = 0.4)

  data.frame(
    y      = y,
    x      = x,
    group  = factor(groups,  levels = c("A", "B", "C")),
    cohort = factor(cohorts, levels = c("Y", "Z"))
  )
}

# ---------------------------------------------------------------------------
# Fit + dump model artefacts (shared across cases for one model)
# ---------------------------------------------------------------------------
fit_and_dump_model <- function(out_dir, data) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  m <- mgcv::gam(y ~ group + cohort + s(x, by = group),
                 data = data, method = "REML")
  write.csv(data, file.path(out_dir, "data.csv"),
            row.names = FALSE, na = "NA")
  beta <- coef(m)
  write.csv(data.frame(name = names(beta), value = unname(beta)),
            file.path(out_dir, "beta.csv"), row.names = FALSE)
  Vp <- vcov(m, freq = FALSE, unconditional = FALSE)
  Vc <- vcov(m, freq = FALSE, unconditional = TRUE)
  write_mm(Vp, file.path(out_dir, "Vp.mtx"))
  write_mm(Vc, file.path(out_dir, "Vc.mtx"))
  m
}

# ---------------------------------------------------------------------------
# Run a single get_difference case and dump
# ---------------------------------------------------------------------------
run_case <- function(case_id, model, comp, cond, f, sim.ci, rm.ranef,
                     n.sim.seed = 1L) {
  out_dir <- file.path(OUT_ROOT, case_id)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  # Pin RNG only for sim.ci paths so the snapshot is reproducible
  if (isTRUE(sim.ci)) set.seed(n.sim.seed)

  res <- itsadug::get_difference(
    model,
    comp     = comp,
    cond     = cond,
    f        = f,
    sim.ci   = sim.ci,
    rm.ranef = rm.ranef,
    se       = TRUE,
    print.summary = FALSE
  )
  write.csv(res, file.path(out_dir, "diff_table.csv"),
            row.names = FALSE, na = "NA")

  # Dump the deterministic sim.ci pieces alongside the RNG-dependent column.
  # se.fit = sqrt(rowSums((p %*% Vc) * p)) — the simultaneous SE has zero
  # RNG variance and gives the Python test a tight anchor; the empirical
  # crit (5% RNG-jittered) is dumped as a scalar for a loose comparison.
  if (isTRUE(sim.ci)) {
    Vc <- vcov(model, freq = FALSE, unconditional = TRUE)
    # Reconstruct p from the same grids itsadug builds internally — see
    # predict.R lines 162–179. We mirror them verbatim so the snapshot
    # corresponds exactly to itsadug's internal lpmatrix difference.
    su <- model$var.summary
    new1 <- list(); new2 <- list()
    for (nm in names(su)) {
      if (nm %in% names(comp)) {
        new1[[nm]] <- comp[[nm]][1]; new2[[nm]] <- comp[[nm]][2]
      } else if (nm %in% names(cond)) {
        new1[[nm]] <- cond[[nm]];    new2[[nm]] <- cond[[nm]]
      } else if (inherits(su[[nm]], "factor")) {
        new1[[nm]] <- as.character(su[[nm]][1]); new2[[nm]] <- as.character(su[[nm]][1])
      } else {
        new1[[nm]] <- su[[nm]][2]; new2[[nm]] <- su[[nm]][2]
      }
    }
    p1 <- mgcv::predict.gam(model, expand.grid(new1), type = "lpmatrix")
    p2 <- mgcv::predict.gam(model, expand.grid(new2), type = "lpmatrix")
    p  <- p1 - p2
    se_fit <- sqrt(rowSums((p %*% Vc) * p))
    # crit = sim.CI / se.fit on any row — equivalent to itsadug's quantile
    # since sim.CI = crit * se.fit. Use the row with the largest se_fit
    # so we don't divide by something near zero.
    j_anchor <- which.max(se_fit)
    crit <- res$sim.CI[j_anchor] / se_fit[j_anchor]
    write.csv(data.frame(se_fit = se_fit),
              file.path(out_dir, "se_fit.csv"), row.names = FALSE)
    write.csv(data.frame(crit = crit),
              file.path(out_dir, "crit.csv"), row.names = FALSE)
  }

  # Args record — Python test consumes this so the fixture is self-describing
  args <- list(
    case_id  = case_id,
    formula  = "y ~ group + cohort + s(x, by = group)",
    comp     = lapply(comp, as.list),
    cond     = lapply(cond, function(v) {
      if (is.numeric(v)) as.list(unname(v)) else as.list(as.character(v))
    }),
    f        = f,
    sim.ci   = sim.ci,
    rm.ranef = if (is.logical(rm.ranef)) rm.ranef else as.character(rm.ranef),
    n_grid   = nrow(res),
    has_sim_ci_col = "sim.CI" %in% colnames(res),
    sim_seed = if (isTRUE(sim.ci)) n.sim.seed else NA_integer_
  )
  write_json <- function(obj, path) {
    cat(toJSON(obj, pretty = TRUE, auto_unbox = TRUE, null = "null"),
        file = path)
  }
  write_json(args, file.path(out_dir, "args.json"))

  # Echo so the run log is self-explanatory
  cat(sprintf("  %-30s  rows=%d  cols=%s\n",
              case_id, nrow(res), paste(colnames(res), collapse = ",")))
  invisible(res)
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
cat("Building itsadug::get_difference fixtures into ", OUT_ROOT, "\n", sep = "")

data <- make_data()
model_dir <- file.path(OUT_ROOT, "_model")
m <- fit_and_dump_model(model_dir, data)

# Grid for the smooth axis. 41 points keeps fixtures small but spans the
# data range (which sits in [0, 10] by construction).
x_grid <- seq(0.5, 9.5, length.out = 41)

# ---- case_basic: 95% pointwise CI, A vs B, no sim.ci ---------------------
run_case(
  "case_basic",
  model    = m,
  comp     = list(group = c("A", "B")),
  cond     = list(x = x_grid),
  f        = 1.96,
  sim.ci   = FALSE,
  rm.ranef = TRUE
)

# ---- case_AC: A vs C, exercises a comp pair with one all-zero smooth -----
run_case(
  "case_AC",
  model    = m,
  comp     = list(group = c("A", "C")),
  cond     = list(x = x_grid),
  f        = 1.96,
  sim.ci   = FALSE,
  rm.ranef = TRUE
)

# ---- case_f99: 99% pointwise CI, f=2.58 ----------------------------------
run_case(
  "case_f99",
  model    = m,
  comp     = list(group = c("A", "B")),
  cond     = list(x = x_grid),
  f        = 2.58,
  sim.ci   = FALSE,
  rm.ranef = TRUE
)

# ---- case_cohortY: hold cohort at user-supplied "Y" via cond -------------
run_case(
  "case_cohortY",
  model    = m,
  comp     = list(group = c("A", "B")),
  cond     = list(x = x_grid, cohort = "Y"),
  f        = 1.96,
  sim.ci   = FALSE,
  rm.ranef = TRUE
)

# ---- case_simci: 95% simultaneous + pointwise, A vs B --------------------
# RNG-seeded so the diff_table.csv is reproducible. The Python test
# treats sim.CI / crit with a loose tol (cross-RNG comparison).
run_case(
  "case_simci",
  model    = m,
  comp     = list(group = c("A", "B")),
  cond     = list(x = x_grid),
  f        = 1.96,
  sim.ci   = TRUE,
  rm.ranef = TRUE,
  n.sim.seed = 12345L
)

cat("Done.\n")
