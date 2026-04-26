#!/usr/bin/env Rscript
# Replicates Lindeløv (2019) "Common statistical tests are linear models"
# data-generation chunks. Run from the repo root:
#
#   Rscript example/stats-data.R
#
# Outputs:
#   example/data/stats/main.csv   — y, x, y2  (matches the post exactly)
#   example/data/stats/anova.csv  — value, group, age, mood (reproducible, but
#     diverges from the post's tables because the post's ggplot geom_jitter
#     consumes RNG state during knitting, which we don't replicate here).

set.seed(40)
rnorm_fixed <- function(N, mu = 0, sd = 1) as.numeric(scale(rnorm(N)) * sd + mu)

OUT <- "example/data/stats"

# ── Wide-format data (post sec 1-8). Lines 174-176 of index.Rmd. ─────────────
y  <- c(rnorm(15), exp(rnorm(15)), runif(20, min = -3, max = 0))
x  <- rnorm_fixed(50, mu = 0, sd = 1)
y2 <- rnorm_fixed(50, mu = 0.5, sd = 1.5)

write.csv(data.frame(y = y, x = x, y2 = y2),
          file.path(OUT, "main.csv"), row.names = FALSE)

# ── Visible plot chunks that consume RNG between sec 1-8 and sec 9-12. ──────
# (Discarded; we only run them so the RNG sequence stays close to the post.)
.D_corr  <- MASS::mvrnorm(30, mu = c(0.9, 0.9),                      # line 197
                          Sigma = matrix(c(1, 0.8, 1, 0.8), ncol = 2),
                          empirical = TRUE)
.D_t1_y  <- rnorm_fixed(20, 0.5, 0.6)                                # line 351
.D_t1_x  <- runif(20, 0.93, 1.07)                                    # line 352
.start   <- rnorm_fixed(20, 0.2, 0.3)                                # line 461
.D_t2_y  <- c(rnorm_fixed(20, 0.3, 0.3), rnorm_fixed(20, 1.3, 0.3))  # line 581
.D_aov_p <- c(rnorm_fixed(15, 0.5, 0.3), rnorm_fixed(15, 0, 0.3),    # line 738
              rnorm_fixed(15, 1, 0.3),   rnorm_fixed(15, 0.8, 0.3))

# ── ANOVA / Kruskal / two-way / ANCOVA data (post sec 9-12). ─────────────────
N <- 20
D <- data.frame(
  value = c(rnorm_fixed(N, 0), rnorm_fixed(N, 1), rnorm_fixed(N, 0.5)),
  group = c(rep("a", N), rep("b", N), rep("c", N))
)
D$group_b <- ifelse(D$group == "b", 1, 0)
D$group_c <- ifelse(D$group == "c", 1, 0)

D$mood <- rep(c("happy", "sad"), nrow(D) / 2)
D$mood_happy <- ifelse(D$mood == "happy", 1, 0)

D$age <- D$value + rnorm_fixed(nrow(D), sd = 3)

write.csv(D[, c("value", "group", "group_b", "group_c",
                "mood", "mood_happy", "age")],
          file.path(OUT, "anova.csv"), row.names = FALSE)

cat("Wrote main.csv (", nrow(data.frame(y)), " rows) and ",
    "anova.csv (", nrow(D), " rows) to ", OUT, "/\n", sep = "")
