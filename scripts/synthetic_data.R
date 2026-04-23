#!/usr/bin/env Rscript
# Synthetic data generators referenced by corpus/feature_matrix.yaml
# meta.synthetic_generators. Writes reproducible CSVs to datasets/synthetic/.
#
# Each generator is seeded; output is idempotent. Re-run to regenerate.

suppressPackageStartupMessages({
  library(stats)
})

OUT <- "datasets/synthetic"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

write_csv <- function(df, name) {
  path <- file.path(OUT, paste0(name, ".csv"))
  write.csv(df, path, row.names = FALSE, na = "NA")
  cat(sprintf("  wrote %-28s  n=%4d  cols=%s\n", basename(path), nrow(df),
              paste(names(df), collapse = ",")))
}

# ---- 1. basic: 5 numeric, 1 factor (3 lvl), 1 group (5 lvl) -----------------
gen_basic <- function() {
  set.seed(1)
  n <- 200
  X <- matrix(rnorm(n * 5), n, 5)
  colnames(X) <- paste0("x", 1:5)
  f <- factor(sample(c("a", "b", "c"), n, TRUE), levels = c("a", "b", "c"))
  g <- factor(sample(paste0("g", 1:5), n, TRUE))
  y <- 1 + 0.5 * X[,1] - 0.3 * X[,2] + ifelse(f == "b", 1, 0) +
       ifelse(f == "c", -1, 0) + rnorm(n, sd = 0.5)
  data.frame(y = y, X, f = f, g = g)
}

# ---- 2. aliased: two perfectly collinear columns -----------------------------
gen_aliased <- function() {
  set.seed(2)
  n <- 100
  x1 <- rnorm(n)
  x2 <- 2 * x1                      # exact linear dependence
  x3 <- rnorm(n)
  f <- factor(sample(c("a", "b", "c"), n, TRUE))
  y <- 1 + x1 + 0.3 * x3 + rnorm(n, sd = 0.4)
  data.frame(y = y, x1 = x1, x2 = x2, x3 = x3, f = f)
}

# ---- 3. empty_cell: factor:factor with one combination absent ----------------
gen_empty_cell <- function() {
  set.seed(3)
  # build a 3x3 grid but drop combination (f=c, g=c)
  combos <- expand.grid(f = c("a", "b", "c"), g = c("a", "b", "c"),
                        stringsAsFactors = FALSE)
  combos <- combos[!(combos$f == "c" & combos$g == "c"), ]
  df <- combos[rep(seq_len(nrow(combos)), each = 15), ]
  df$x <- rnorm(nrow(df))
  df$y <- 1 + df$x + as.integer(factor(df$f)) * 0.3 +
          as.integer(factor(df$g)) * -0.2 + rnorm(nrow(df), sd = 0.4)
  df$f <- factor(df$f, levels = c("a", "b", "c"))
  df$g <- factor(df$g, levels = c("a", "b", "c"))
  df[, c("y", "x", "f", "g")]
}

# ---- 4. singleton: factor level with 1 obs, plus unused level ----------------
gen_singleton <- function() {
  set.seed(4)
  n <- 100
  # 49 a, 50 b, 1 c; unused level 'd' declared but never sampled
  f <- factor(c(rep("a", 49), rep("b", 50), "c"),
              levels = c("a", "b", "c", "d"))
  f <- f[sample(seq_len(n))]
  x <- rnorm(n)
  y <- 1 + x + ifelse(f == "b", 1, 0) + rnorm(n, sd = 0.4)
  data.frame(y = y, x = x, f = f)
}

# ---- 5. big_factor: 50 balanced levels ---------------------------------------
gen_big_factor <- function() {
  set.seed(5)
  n <- 500
  g <- factor(rep(paste0("g", sprintf("%02d", 1:50)), length.out = n))
  x <- rnorm(n)
  u <- rnorm(nlevels(g), sd = 0.8)            # random intercept per level
  y <- 1 + 0.5 * x + u[g] + rnorm(n, sd = 0.3)
  data.frame(y = y, x = x, g = g)
}

# ---- 6. na_mixed: ~5% NA across several columns ------------------------------
gen_na_mixed <- function() {
  set.seed(6)
  n <- 200
  df <- gen_basic()[seq_len(n), ]                # reuse basic structure
  inject <- function(v, p = 0.05) {
    v[sample(length(v), ceiling(length(v) * p))] <- NA
    v
  }
  df$x1 <- inject(df$x1)
  df$x3 <- inject(df$x3)
  df$f  <- inject(df$f)
  df$y  <- inject(df$y, 0.03)
  df
}

# ---- 7. nested_group: g2 within g1, imbalanced -------------------------------
gen_nested_group <- function() {
  set.seed(7)
  # 5 top-level groups, 2-5 subgroups each, 10-30 rows per subgroup
  rows <- list()
  idx <- 1
  for (i in 1:5) {
    n_sub <- sample(2:5, 1)
    for (j in seq_len(n_sub)) {
      n_rows <- sample(10:30, 1)
      x <- rnorm(n_rows)
      y <- 1 + 0.4 * x + i * 0.2 + j * 0.1 + rnorm(n_rows, sd = 0.3)
      rows[[idx]] <- data.frame(
        y  = y,
        x  = x,
        g1 = factor(paste0("G", i)),
        g2 = factor(paste0("G", i, ".", j))
      )
      idx <- idx + 1
    }
  }
  do.call(rbind, rows)
}

# ---- 8. crossed_re: two fully-crossed grouping factors -----------------------
gen_crossed_re <- function() {
  set.seed(8)
  grid <- expand.grid(g1 = paste0("A", 1:8), g2 = paste0("B", 1:10),
                      stringsAsFactors = FALSE)
  df <- grid[rep(seq_len(nrow(grid)), each = 5), ]  # 8*10*5 = 400
  df$x  <- rnorm(nrow(df))
  u1 <- rnorm(8,  sd = 0.7); names(u1) <- paste0("A", 1:8)
  u2 <- rnorm(10, sd = 0.5); names(u2) <- paste0("B", 1:10)
  df$y <- 1 + 0.5 * df$x + u1[df$g1] + u2[df$g2] + rnorm(nrow(df), sd = 0.4)
  df$g1 <- factor(df$g1); df$g2 <- factor(df$g2)
  df[, c("y", "x", "g1", "g2")]
}

# ---- 9. longitudinal: 30 subjects x 10 timepoints, RE intercept+slope --------
gen_longitudinal <- function() {
  set.seed(9)
  n_sub <- 30; n_t <- 10
  sub <- rep(seq_len(n_sub), each = n_t)
  t   <- rep(seq_len(n_t) - 1, n_sub)           # 0..9
  b0  <- rnorm(n_sub, sd = 2)
  b1  <- rnorm(n_sub, sd = 0.5)
  y <- 5 + 1.5 * t + b0[sub] + b1[sub] * t + rnorm(length(sub), sd = 0.6)
  data.frame(y = y, time = t, id = factor(sprintf("S%02d", sub)))
}

# ---- 10. 2d_smooth: x,y in [0,1]^2 with known surface ------------------------
gen_2d_smooth <- function() {
  set.seed(10)
  n <- 400
  x <- runif(n); y <- runif(n)
  z <- sin(2 * pi * x) * cos(pi * y) + 0.3 * x - 0.2 * y +
       rnorm(n, sd = 0.2)
  data.frame(z = z, x = x, y = y)
}

# ---- run all -----------------------------------------------------------------
cat("Generating synthetic datasets in", OUT, "\n")
write_csv(gen_basic(),         "seed_synth_basic")
write_csv(gen_aliased(),       "seed_synth_aliased")
write_csv(gen_empty_cell(),    "seed_synth_empty_cell")
write_csv(gen_singleton(),     "seed_synth_singleton")
write_csv(gen_big_factor(),    "seed_synth_big_factor")
write_csv(gen_na_mixed(),      "seed_synth_na_mixed")
write_csv(gen_nested_group(),  "seed_synth_nested_group")
write_csv(gen_crossed_re(),    "seed_synth_crossed_re")
write_csv(gen_longitudinal(),  "seed_synth_longitudinal")
write_csv(gen_2d_smooth(),     "seed_synth_2d_smooth")
cat("done.\n")
