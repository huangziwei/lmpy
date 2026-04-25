#!/usr/bin/env Rscript
# Generate stats::glm() oracle outputs for the lmpy.glm parity tests.
#
# Reads:
#   datasets/<pkg>/<name>.csv
# Writes:
#   tests/fixtures/glm/<id>/oracle.json   — pinned numerical outputs of glm()
#
# Tests load these via tests/conftest.py:load_glm_oracle(). Each fixture
# corresponds to one (dataset, formula, family, link) triple from the plan
# (.claude/plans/glm-port.md, Phase 0.2).

suppressPackageStartupMessages({
  library(jsonlite)
  library(MASS)         # quine, menarche, Insurance
})

OUT_ROOT  <- "tests/fixtures/glm"
DATA_ROOT <- "datasets"
dir.create(OUT_ROOT, showWarnings = FALSE, recursive = TRUE)

# --------------------------------------------------------------------------
# Dataset loading — same CSV layout as scripts/make_fixtures.R reads.
# Factor columns: read as character then convert based on a small per-dataset
# whitelist below (we don't need the full schema-json round-trip the GAM
# fixture-maker uses; glm() only cares that columns are factors of the right
# levels, not that the contrast labels match exactly).
# --------------------------------------------------------------------------
load_csv <- function(pkg, name) {
  pkg_dir <- if (pkg == "datasets") "R" else pkg
  read.csv(file.path(DATA_ROOT, pkg_dir, paste0(name, ".csv")),
           stringsAsFactors = FALSE, check.names = FALSE)
}

prep_quine <- function() {
  d <- load_csv("MASS", "quine")
  d$Eth <- factor(d$Eth, levels = c("A", "N"))
  d$Sex <- factor(d$Sex, levels = c("F", "M"))
  d$Age <- factor(d$Age, levels = c("F0", "F1", "F2", "F3"))
  d$Lrn <- factor(d$Lrn, levels = c("AL", "SL"))
  d
}
prep_insurance <- function() {
  d <- load_csv("MASS", "Insurance")
  # MASS::Insurance: District = unordered factor; Group, Age = ordered.
  # Mirrors the dataset's own schema (and lmpy's datasets/MASS/Insurance.schema.json).
  # R glm() uses contr.poly for ordered factors by default; lmpy applies
  # the same poly contrasts when the column is registered ordered via
  # the conftest's `set_ordered_cols`.
  d$District <- factor(d$District, levels = c("1", "2", "3", "4"))
  d$Group <- factor(d$Group, levels = c("<1l", "1-1.5l", "1.5-2l", ">2l"),
                    ordered = TRUE)
  d$Age <- factor(d$Age, levels = c("<25", "25-29", "30-35", ">35"),
                  ordered = TRUE)
  d
}
prep_menarche <- function() {
  d <- load_csv("MASS", "menarche")
  # Already numeric; cbind() form below uses Total - Menarche directly.
  d
}
prep_iris <- function() {
  d <- load_csv("R", "iris")
  d$Species <- factor(d$Species,
                      levels = c("setosa", "versicolor", "virginica"))
  d
}
prep_trees <- function() {
  load_csv("R", "trees")
}

# --------------------------------------------------------------------------
# Cases: one entry per oracle.
# --------------------------------------------------------------------------
cases <- list(
  list(id      = "gaussian_identity_iris",
       data    = prep_iris(),
       formula = Sepal.Length ~ Petal.Length + Species,
       family  = gaussian(link = "identity"),
       offset  = NULL),

  list(id      = "gaussian_log_insurance",
       data    = local({
         d <- prep_insurance()
         d[d$Claims > 0, ]   # log-link needs μ>0; drop the one Claims==0 row
       }),
       # Non-canonical Gaussian + offset. Holders > 0 always.
       formula = Claims ~ District + Group,
       family  = gaussian(link = "log"),
       offset  = quote(log(Holders)),
       start   = "log_y"),     # start = log(y), η-scale start for log link

  list(id      = "gamma_inverse_trees",
       data    = prep_trees(),
       formula = Volume ~ log(Height) + log(Girth),
       family  = Gamma(link = "inverse"),
       offset  = NULL),

  list(id      = "gamma_log_trees",
       data    = prep_trees(),
       formula = Volume ~ log(Height) + log(Girth),
       family  = Gamma(link = "log"),
       offset  = NULL),

  list(id      = "poisson_log_quine",
       data    = prep_quine(),
       formula = Days ~ Sex + Age + Eth + Lrn,
       family  = poisson(link = "log"),
       offset  = NULL),

  list(id      = "poisson_sqrt_quine",
       data    = prep_quine(),
       formula = Days ~ Sex + Age + Eth + Lrn,
       family  = poisson(link = "sqrt"),
       offset  = NULL),

  list(id      = "binomial_logit_menarche",
       data    = prep_menarche(),
       formula = cbind(Menarche, Total - Menarche) ~ Age,
       family  = binomial(link = "logit"),
       offset  = NULL),

  list(id      = "binomial_probit_menarche",
       data    = prep_menarche(),
       formula = cbind(Menarche, Total - Menarche) ~ Age,
       family  = binomial(link = "probit"),
       offset  = NULL),

  list(id      = "binomial_cauchit_menarche",
       data    = prep_menarche(),
       formula = cbind(Menarche, Total - Menarche) ~ Age,
       family  = binomial(link = "cauchit"),
       offset  = NULL),

  list(id      = "binomial_cloglog_menarche",
       data    = prep_menarche(),
       formula = cbind(Menarche, Total - Menarche) ~ Age,
       family  = binomial(link = "cloglog"),
       offset  = NULL),

  list(id      = "ig_canonical_insurance",
       data    = local({
         d <- prep_insurance()
         d[d$Claims > 0, ]   # IG requires y>0 strictly
       }),
       formula = Claims ~ Group,
       family  = inverse.gaussian(link = "1/mu^2"),
       offset  = NULL)
)

# --------------------------------------------------------------------------
# Fit + serialize
# --------------------------------------------------------------------------
fit_and_dump <- function(case) {
  d <- case$data
  args <- list(formula = case$formula, data = d, family = case$family)
  if (!is.null(case$offset)) args$offset <- eval(case$offset, envir = d)
  if (!is.null(case$start) && case$start == "log_y") {
    # Build a feasible β̂_0 by fitting log(y) on the same X with lm; gives R
    # a η-scale start that satisfies link.linkinv(η)>0 for the log link.
    y <- model.response(model.frame(case$formula, data = d))
    m_start <- lm(log(y + 1e-3) ~ ., data = model.frame(case$formula, data = d)[-1])
    args$start <- coef(m_start)
  }
  m <- do.call(glm, args)
  s <- summary(m)

  # `predict(..., se.fit=TRUE)` on the fitted X: returns fit (η̂) and se.fit
  # for type="link"; we do response separately so the delta-method check has
  # both sides pinned.
  pred_link <- predict(m, type = "link", se.fit = TRUE)
  pred_resp <- predict(m, type = "response", se.fit = TRUE)

  # 95% Wald CI from confint.default — this is the lm-style CI lmpy returns
  # (NOT the profile-likelihood confint.glm).
  ci <- suppressWarnings(confint.default(m, level = 0.95))

  out <- list(
    id              = case$id,
    formula         = deparse1(case$formula),
    family_name     = m$family$family,
    link_name       = m$family$link,
    n               = as.integer(nrow(d)),
    coef_names      = names(coef(m)),
    coefficients    = unname(coef(m)),
    std_error       = unname(s$coefficients[, "Std. Error"]),
    # column 3 of summary$coefficients is "z value" or "t value" depending on
    # family.scale_known — name it generically and let the test choose.
    test_stat       = unname(s$coefficients[, 3]),
    p_value         = unname(s$coefficients[, 4]),
    test_kind       = ifelse(colnames(s$coefficients)[3] == "z value", "z", "t"),
    ci_lower        = unname(ci[, 1]),
    ci_upper        = unname(ci[, 2]),
    vcov            = unname(as.matrix(vcov(m))),
    deviance        = as.numeric(m$deviance),
    null_deviance   = as.numeric(m$null.deviance),
    df_residual     = as.integer(m$df.residual),
    df_null         = as.integer(m$df.null),
    aic             = as.numeric(m$aic),
    bic             = as.numeric(BIC(m)),
    loglik          = as.numeric(logLik(m)),
    loglik_df       = attr(logLik(m), "df"),
    dispersion      = as.numeric(s$dispersion),
    iter            = as.integer(m$iter),
    converged       = as.logical(m$converged),
    fitted_values   = unname(fitted(m)),
    linear_pred     = unname(m$linear.predictors),
    res_deviance    = unname(residuals(m, type = "deviance")),
    res_pearson     = unname(residuals(m, type = "pearson")),
    res_working     = unname(residuals(m, type = "working")),
    res_response    = unname(residuals(m, type = "response")),
    pred_link_fit   = unname(pred_link$fit),
    pred_link_se    = unname(pred_link$se.fit),
    pred_resp_fit   = unname(pred_resp$fit),
    pred_resp_se    = unname(pred_resp$se.fit)
  )

  out_dir <- file.path(OUT_ROOT, case$id)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  write_json(out, file.path(out_dir, "oracle.json"),
             auto_unbox = TRUE, digits = 17, matrix = "rowmajor",
             pretty = FALSE)
  cat(sprintf("  wrote %s (%d obs, %d coefs, dev=%.6g)\n",
              case$id, out$n, length(out$coefficients), out$deviance))
}

cat(sprintf("Generating %d glm() oracles...\n", length(cases)))
for (case in cases) {
  res <- tryCatch(fit_and_dump(case), error = function(e) {
    cat(sprintf("  FAILED %s: %s\n", case$id, conditionMessage(e)))
    NULL
  })
}
cat("done.\n")
