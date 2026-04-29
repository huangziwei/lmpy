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
from .dataframe import DataFrame, GroupBy, desc, tbl
from . import plot

# Expose hea's free-function data-transform helpers on the polars
# namespace so chains and prep code can stay in one import:
# ``import polars as pl`` is enough, no separate ``hea.*`` namespace
# inside ``arrange()`` / column casts. Verbs themselves stay as
# methods on the DataFrame subclass (no namespace needed).
#
# Monkey-patching upstream is normally bad form. We accept it only
# for symbols polars hasn't claimed (``desc``, ``tbl``, ``factor``)
# and refuse to overwrite if polars later adds its own — caller gets
# a clear error rather than silently changed semantics. We do NOT
# patch ``pl.data`` (R-specific loader, not a polars-y concept) or
# the ``DataFrame`` / ``GroupBy`` classes (would conflict with
# ``pl.DataFrame`` which has its own meaning, including via
# ``pl.read_csv`` and other polars constructors).
import polars as _pl
for _name, _obj in [("desc", desc), ("tbl", tbl), ("factor", factor)]:
    if hasattr(_pl, _name) and getattr(_pl, _name) is not _obj:
        raise RuntimeError(
            f"polars now exports its own `pl.{_name}`; hea would shadow it. "
            "Update hea to use the upstream version."
        )
    setattr(_pl, _name, _obj)
del _pl, _name, _obj
