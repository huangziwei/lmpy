from .compare import AIC, BIC, anova
from .family import (
    Binomial,
    Family,
    Gamma,
    Gaussian,
    InverseGaussian,
    Poisson,
    binomial,
    gaussian,
    inverse_gaussian,
    poisson,
)
from .gam import gam
from .glm import glm
from .lm import lm
from .lme import lme
from .stats import (
    aov,
    chisq_test,
    cor_test,
    kruskal_test,
    rank,
    signed_rank,
    t_test,
    wilcox_test,
)
from .data import data, factor
