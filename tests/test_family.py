"""mgcv-oracle regression tests for hea.family.

Pins per-link and per-family numerics against the canonical R/mgcv values
generated with stats::make.link / stats::<family>() / mgcv:::fix.family.*
at fixed (μ, η, scale) inputs. Test points cover the boundary of each
link's domain so that overflow/underflow paths get exercised too.
"""

from __future__ import annotations

import numpy as np
import pytest

from hea.family import (
    Binomial,
    CauchitLink,
    CloglogLink,
    Gamma,
    Gaussian,
    InverseGaussian,
    InverseSquareLink,
    LogitLink,
    Poisson,
    ProbitLink,
    SqrtLink,
    Tweedie,
    tw,
)


MUS = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
ETAS = np.array([-2.5, -0.5, 0.0, 0.7, 2.0])


# ---------------------------------------------------------------------------
# Links — values pinned to R::stats::make.link + mgcv:::fix.family.link.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,link_oracle,linkinv_oracle,mu_eta_oracle,d2_oracle,d3_oracle,d4_oracle",
    [
        (
            LogitLink,
            [-2.9444389791664403, -1.3862943611198906, 0.0, 1.3862943611198908, 2.9444389791664394],
            [0.07585818002124356, 0.37754066879814546, 0.5, 0.66818777216816616, 0.88079707797788243],
            [0.070103716545108177, 0.23500371220159449, 0.25, 0.22171287329310907, 0.10499358540350652],
            [-398.89196675900268, -23.437499999999996, 0.0, 23.437500000000014, 398.89196675900212],
            [16002.33270155999, 253.90624999999994, 32.0, 253.90625000000017, 16002.332701559952],
            [-959992.63357402082, -3735.3515624999991, 0.0, 3735.3515625000032, 959992.63357401767],
        ),
        (
            ProbitLink,
            [-1.6448536269514726, -0.84162123357291418, 0.0, 0.84162123357291441, 1.6448536269514715],
            [0.0062096653257761349, 0.30853753872598694, 0.5, 0.75803634777692697, 0.97724986805182079],
            [0.01752830049356854, 0.35206532676429952, 0.3989422804014327, 0.31225393336676127, 0.053990966513188063],
            [-154.6356833289386, -10.737885188829354, 0.0, 10.737885188829367, 154.63568332893792],
            [5843.9347164857427, 110.1329652673685, 15.749609945722417, 110.13296526736866, 5843.9347164857027],
            [-337755.43402393488, -1541.2451459794413, 0.0, 1541.2451459794445, 337755.4340239318],
        ),
        (
            CauchitLink,
            [-6.3137515146750438, -1.3763819204711736, 0.0, 1.3763819204711742, 6.3137515146750376],
            [0.12111894159084341, 0.35241638234956674, 0.5, 0.69440011221421472, 0.85241638234956674],
            [0.043904811887419404, 0.25464790894703254, 0.31830988618379069, 0.21363079609650382, 0.063661977236758135],
            [-5092.749842851993, -78.637795426380578, 0.0, 78.637795426380677, 5092.7498428519775],
            [305581.72289978294, 1199.5876940407713, 62.012553360599632, 1199.5876940407732, 305581.72289978171],
            [-24446195.30522709, -23852.714815240906, 0.0, 23852.71481524095, 24446195.30522697],
        ),
        (
            CloglogLink,
            [-2.9701952490421637, -1.4999399867595158, -0.36651292058166435, 0.4758849953271107, 1.0971887003649483],
            [0.07880634482448419, 0.45476078810739495, 0.63212055882855767, 0.86651320334191617, 0.99938202101066886],
            [0.075616179917426515, 0.33070429889041808, 0.36787944117144233, 0.26880939818177735, 0.0045662814201279153],
            [-399.54304335262657, -24.377665865346501, -2.5546957604665774, 5.8819458413853711, 88.952114337550896],
            [16000.865329326627, 251.39709691607473, 21.17475642459932, 70.530011682474793, 3261.7900829111441],
            [-959997.46017832356, -3745.1331057436109, -67.169617126383642, 915.99384298698556, 183838.66891563043],
        ),
        (
            SqrtLink,
            [0.22360679774997896, 0.44721359549995793, 0.70710678118654757, 0.89442719099991586, 0.97467943448089633],
            [6.25, 0.25, 0.0, 0.48999999999999994, 4.0],
            [-5.0, -1.0, 0.0, 1.3999999999999999, 4.0],
            [-22.360679774997894, -2.7950849718747368, -0.70710678118654757, -0.3493856214843421, -0.26999430318030371],
            [670.82039324993684, 20.963137289060526, 2.1213203435596428, 0.65509804028314145, 0.42630679449521647],
            [-33541.019662496838, -262.03921611325654, -10.606601717798213, -2.0471813758848167, -1.1218599855137275],
        ),
        (
            InverseSquareLink,
            [400.0, 25.0, 4.0, 1.5624999999999998, 1.10803324099723],
            # linkinv at η<0 is NaN in R; we only check the η>0 entries.
            [np.nan, np.nan, np.nan, 1.1952286093343936, 0.70710678118654746],
            [np.nan, np.nan, np.nan, -0.8537347209531384, -0.17677669529663687],
            [959999.99999999977, 3749.9999999999991, 96.0, 14.648437499999996, 7.3664259789289535],
            [-76799999.99999997, -74999.999999999971, -768.0, -73.242187499999972, -31.016530437595598],
            [7679999999.9999971, 1874999.9999999993, 7680.0, 457.76367187499983, 163.24489703997685],
        ),
    ],
)
def test_link_values_match_mgcv(
    cls, link_oracle, linkinv_oracle, mu_eta_oracle, d2_oracle, d3_oracle, d4_oracle,
):
    lk = cls()
    np.testing.assert_allclose(lk.link(MUS), link_oracle, rtol=1e-12, atol=0,
                               err_msg=f"{lk.name}.link")
    np.testing.assert_allclose(lk.d2link(MUS), d2_oracle, rtol=1e-12, atol=0,
                               err_msg=f"{lk.name}.d2link")
    np.testing.assert_allclose(lk.d3link(MUS), d3_oracle, rtol=1e-12, atol=0,
                               err_msg=f"{lk.name}.d3link")
    np.testing.assert_allclose(lk.d4link(MUS), d4_oracle, rtol=1e-12, atol=0,
                               err_msg=f"{lk.name}.d4link")
    # InverseSquare's linkinv/mu_eta are only defined for η>0; mask the rest.
    linkinv_oracle = np.asarray(linkinv_oracle, dtype=float)
    mu_eta_oracle = np.asarray(mu_eta_oracle, dtype=float)
    mask = ~np.isnan(linkinv_oracle)
    np.testing.assert_allclose(lk.linkinv(ETAS[mask]), linkinv_oracle[mask],
                               rtol=1e-12, atol=0, err_msg=f"{lk.name}.linkinv")
    np.testing.assert_allclose(lk.mu_eta(ETAS[mask]), mu_eta_oracle[mask],
                               rtol=1e-12, atol=0, err_msg=f"{lk.name}.mu_eta")


@pytest.mark.parametrize("cls", [LogitLink, ProbitLink, CauchitLink, CloglogLink, SqrtLink])
def test_link_round_trip(cls):
    """linkinv(link(μ)) ≈ μ on the link's natural domain."""
    lk = cls()
    mu = MUS if cls is not SqrtLink else np.array([0.1, 0.4, 1.0, 2.5, 9.0])
    np.testing.assert_allclose(lk.linkinv(lk.link(mu)), mu, rtol=1e-12, atol=0)


def test_inverse_square_round_trip():
    lk = InverseSquareLink()
    mu = np.array([0.1, 0.5, 1.0, 2.5, 9.0])
    np.testing.assert_allclose(lk.linkinv(lk.link(mu)), mu, rtol=1e-12, atol=0)


def test_link_valideta():
    # sqrt and 1/μ²  reject η ≤ 0 (matches R make.link).
    assert SqrtLink().valideta(np.array([0.1, 1.0])) is True
    assert SqrtLink().valideta(np.array([0.1, 0.0])) is False
    assert InverseSquareLink().valideta(np.array([0.1, 1.0])) is True
    assert InverseSquareLink().valideta(np.array([0.1, -1.0])) is False
    # Bernoulli-type links accept any finite η.
    assert LogitLink().valideta(np.array([-1e3, 0.0, 1e3])) is True


# ---------------------------------------------------------------------------
# Poisson family — pinned against R::stats::poisson + mgcv::fix.family.{var,ls}.
# ---------------------------------------------------------------------------


def test_poisson_static_fields():
    f = Poisson()
    assert f.name == "poisson"
    assert f.canonical_link_name == "log"
    assert f.scale_known is True
    assert f.is_canonical
    # variance/dvar/d2var
    mu = np.array([0.5, 1.2, 2.1])
    np.testing.assert_array_equal(f.variance(mu), mu)
    np.testing.assert_array_equal(f.dvar(mu), np.ones_like(mu))
    np.testing.assert_array_equal(f.d2var(mu), np.zeros_like(mu))


def test_poisson_oracle():
    f = Poisson()
    y = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    mu = np.array([0.5, 1.2, 2.1, 2.8, 4.5])
    wt = np.array([1.0, 2.0, 1.0, 1.0, 3.0])
    np.testing.assert_allclose(
        f.dev_resids(y, mu, wt),
        [1.0, 0.07071377282418145, 0.00483934332227196,
         0.01395722892170814, 0.16081546973479077],
        rtol=1e-12, atol=0,
    )
    np.testing.assert_allclose(f.aic(y, mu, None, wt, len(y)),
                               21.297689743799772, rtol=1e-12, atol=0)
    np.testing.assert_allclose(f.ls(y, wt, 1.0),
                               [-10.0236819644984, 0.0, 0.0],
                               rtol=1e-12, atol=0)


def test_poisson_initialize():
    f = Poisson()
    y = np.array([0.0, 1.0, 5.0])
    np.testing.assert_allclose(f.initialize(y, np.ones(3)), y + 0.1)
    with pytest.raises(ValueError, match="negative values"):
        f.initialize(np.array([-1.0, 0.0]), np.ones(2))


def test_poisson_validmu():
    assert Poisson().validmu(np.array([0.1, 1.0, 100.0]))
    assert not Poisson().validmu(np.array([0.0, 1.0]))
    assert not Poisson().validmu(np.array([np.inf, 1.0]))


# ---------------------------------------------------------------------------
# Binomial family — pinned against R::stats::binomial + mgcv.
# ---------------------------------------------------------------------------


def test_binomial_static_fields():
    f = Binomial()
    assert f.name == "binomial"
    assert f.canonical_link_name == "logit"
    assert f.scale_known is True
    mu = np.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(f.variance(mu), mu * (1 - mu), rtol=1e-12)
    np.testing.assert_allclose(f.dvar(mu), 1 - 2 * mu, rtol=1e-12)
    np.testing.assert_array_equal(f.d2var(mu), -2.0 * np.ones_like(mu))


def test_binomial_bernoulli_oracle():
    f = Binomial()
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    mu = np.array([0.3, 0.7, 0.6, 0.4, 0.85])
    wt = np.ones(5)
    np.testing.assert_allclose(
        f.dev_resids(y, mu, wt),
        [0.713349887877465, 0.713349887877465, 1.021651247531981,
         1.021651247531981, 0.325037858995550],
        rtol=1e-12, atol=0,
    )
    np.testing.assert_allclose(f.aic(y, mu, None, wt, len(y)),
                               3.79504012981444, rtol=1e-12, atol=0)
    # mgcv ls is identically zero in the Bernoulli case (saturated dbinom = 1).
    np.testing.assert_array_equal(f.ls(y, wt, 1.0), np.zeros(3))


def test_binomial_proportion_oracle():
    """y is the success proportion in [0,1]; wt is the binomial size m."""
    f = Binomial()
    y = np.array([0.2, 0.5, 0.7, 0.0])
    mu = np.array([0.3, 0.4, 0.6, 0.1])
    wt = np.array([5.0, 4.0, 10.0, 3.0])
    np.testing.assert_allclose(
        f.dev_resids(y, mu, wt),
        [0.257320924779854, 0.163287978081021,
         0.432017082870932, 0.632163093946958],
        rtol=1e-12, atol=0,
    )
    np.testing.assert_allclose(f.aic(y, mu, None, wt, len(y)),
                               7.87389855174967, rtol=1e-12, atol=0)
    np.testing.assert_allclose(f.ls(y, wt, 1.0),
                               [-3.19455473603545, 0.0, 0.0],
                               rtol=1e-11, atol=0)


def test_binomial_initialize_and_validmu():
    f = Binomial()
    y = np.array([0.0, 0.5, 1.0])
    wt = np.array([1.0, 3.0, 1.0])
    expected = (wt * y + 0.5) / (wt + 1.0)
    np.testing.assert_allclose(f.initialize(y, wt), expected, rtol=1e-12)
    with pytest.raises(ValueError, match="0 <= y <= 1"):
        f.initialize(np.array([-0.1, 0.5]), np.ones(2))
    assert f.validmu(np.array([0.01, 0.5, 0.99]))
    assert not f.validmu(np.array([0.0, 0.5]))
    assert not f.validmu(np.array([0.5, 1.0]))


def test_binomial_dev_resids_no_warning_on_boundary():
    """y=0,μ=0 and y=1,μ=1 must yield 0 contribution without numpy warnings."""
    f = Binomial()
    with np.errstate(divide="raise", invalid="raise"):
        d = f.dev_resids(
            np.array([0.0, 1.0, 0.5]),
            np.array([1e-15, 1.0 - 1e-15, 0.5]),
            np.ones(3),
        )
    assert np.all(np.isfinite(d))


# ---------------------------------------------------------------------------
# InverseGaussian family — pinned against R::stats::inverse.gaussian + mgcv.
# ---------------------------------------------------------------------------


def test_inverse_gaussian_static_fields():
    f = InverseGaussian()
    assert f.name == "inverse.gaussian"
    assert f.canonical_link_name == "1/mu^2"
    assert f.scale_known is False
    mu = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(f.variance(mu), mu ** 3, rtol=1e-12)
    np.testing.assert_allclose(f.dvar(mu), 3 * mu ** 2, rtol=1e-12)
    np.testing.assert_allclose(f.d2var(mu), 6 * mu, rtol=1e-12)


def test_inverse_gaussian_oracle():
    f = InverseGaussian()
    y = np.array([1.0, 2.0, 0.5, 3.0])
    mu = np.array([0.9, 2.1, 0.6, 2.5])
    wt = np.array([1.0, 1.0, 2.0, 1.0])
    dev_v = f.dev_resids(y, mu, wt)
    np.testing.assert_allclose(
        dev_v,
        [0.01234567901234567, 0.00113378684807256,
         0.11111111111111106, 0.01333333333333333],
        rtol=1e-12, atol=0,
    )
    dev = float(dev_v.sum())
    np.testing.assert_allclose(f.aic(y, mu, dev, wt, len(y)),
                               -0.546674508250539, rtol=1e-12, atol=0)
    # log-scale derivatives: d1 = -nobs/2 = -2; d2 = 0 (algebraic
    # cancellation since the log(2π φ y³) term is linear in log φ).
    np.testing.assert_allclose(f.ls(y, wt, 0.5),
                               [-3.59080461442099, -2.0, 0.0],
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(f.ls(y, wt, 2.0),
                               [-6.36339333666077, -2.0, 0.0],
                               rtol=1e-12, atol=0)


def test_inverse_gaussian_initialize():
    f = InverseGaussian()
    y = np.array([0.5, 1.0, 2.5])
    np.testing.assert_array_equal(f.initialize(y, np.ones(3)), y)
    with pytest.raises(ValueError, match="positive values"):
        f.initialize(np.array([0.0, 1.0]), np.ones(2))


# ---------------------------------------------------------------------------
# Existing families should not regress.  Gamma.ls is now log-scale; pin the
# converted values so a future refactor doesn't silently drift back.
# ---------------------------------------------------------------------------


def test_gaussian_unchanged_oracle():
    # ls(y=μ=[1..4], wt=1, scale): log-scale convention (d1 = -n/2, d2 = 0).
    f = Gaussian()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = f.ls(y, np.ones(4), 1.0)
    np.testing.assert_allclose(out, [-0.5 * 4 * np.log(2 * np.pi), -2.0, 0.0],
                               rtol=1e-12, atol=0)


def test_gaussian_ls_with_weights_oracle():
    """Pins mgcv's gaussian()$ls form: ls0 = -nobs/2·log(2πφ) + ½·Σ log w[w>0],
    d/d log φ = -nobs/2, d²/d log φ² = 0.  ``nobs`` is the count of w>0
    observations, NOT Σw — weights act as precision multipliers, not as
    sample-size multipliers."""
    f = Gaussian()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    wt = np.array([0.0, 1.5, 2.0, 0.5])              # one zero-weight row
    nobs = 3
    log_w_sum = float(np.sum(np.log(wt[wt > 0])))    # log(1.5)+log(2)+log(0.5)
    # scale=0.5
    expected_ls0 = -0.5 * nobs * np.log(2.0 * np.pi * 0.5) + 0.5 * log_w_sum
    np.testing.assert_allclose(
        f.ls(y, wt, 0.5),
        [expected_ls0, -0.5 * nobs, 0.0],
        rtol=1e-12, atol=0,
    )
    # scale=2.0
    expected_ls0 = -0.5 * nobs * np.log(2.0 * np.pi * 2.0) + 0.5 * log_w_sum
    np.testing.assert_allclose(
        f.ls(y, wt, 2.0),
        [expected_ls0, -0.5 * nobs, 0.0],
        rtol=1e-12, atol=0,
    )


def test_gamma_ls_log_scale_conversion():
    """mgcv returns d/dφ; we apply the chain rule to log φ. Pin both scales."""
    f = Gamma()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    wt = np.ones(4)
    # mgcv raw form at scale=0.5: [-5.63287638586838, -4.32580552738365, 8.02744183124810]
    # convert: d1_log = 0.5·(-4.32580552738365) = -2.162902763691825
    #          d2_log = 0.5·(-4.32580552738365) + 0.25·(8.02744183124810)
    #                 = -0.156042305879800
    np.testing.assert_allclose(
        f.ls(y, wt, 0.5),
        [-5.63287638586838, -2.162902763691825, -0.156042305879800],
        rtol=1e-10, atol=0,
    )
    # mgcv raw at scale=2: [-8.853807963166636, -1.270362845461478, 0.536662295325308]
    # d1_log = 2·(-1.270362845461478) = -2.540725690922956
    # d2_log = 2·(-1.270362845461478) + 4·(0.536662295325308) = -0.394076509621724
    np.testing.assert_allclose(
        f.ls(y, wt, 2.0),
        [-8.853807963166636, -2.540725690922956, -0.394076509621724],
        rtol=1e-10, atol=0,
    )


# ---------------------------------------------------------------------------
# Family/link composition behaviour
# ---------------------------------------------------------------------------


def test_family_link_resolution():
    # default link = canonical
    assert Poisson().link.name == "log"
    assert Binomial().link.name == "logit"
    assert InverseGaussian().link.name == "1/mu^2"
    # explicit string
    assert Poisson(link="sqrt").link.name == "sqrt"
    assert Binomial(link="probit").link.name == "probit"
    # is_canonical reports correctly
    assert Poisson().is_canonical
    assert not Poisson(link="sqrt").is_canonical
    assert Binomial().is_canonical
    assert not Binomial(link="cloglog").is_canonical


def test_family_link_unknown_raises():
    with pytest.raises(ValueError, match="unknown link"):
        Poisson(link="banana")


# ---------------------------------------------------------------------------
# Tweedie / tw — oracles pinned to mgcv 1.9-4.
#
# All numeric oracles in this section are produced by R/mgcv:
#   Tweedie(p, link='log')$variance / dev.resids / dvar / d2var / d3var
#   ldTweedie(y, mu, rho=log(phi), theta, a, b)              # log f + derivs
# The (rho, theta) parametrisation matches hea's tw() exactly:
#   p(theta) = (a + b·exp(theta))/(1 + exp(theta));  rho = log(phi).
# ---------------------------------------------------------------------------


def test_tweedie_static_fields():
    f = Tweedie(p=1.5)
    assert f.name == "Tweedie"
    assert f.canonical_link_name == "log"
    assert f.scale_known is False
    assert f.link.name == "log"
    assert f.p == 1.5
    assert f.n_theta == 0  # fixed-p Tweedie isn't "estimable"


def test_tweedie_p_out_of_range_raises():
    with pytest.raises(ValueError, match="1 < p < 2"):
        Tweedie(p=1.0)
    with pytest.raises(ValueError, match="1 < p < 2"):
        Tweedie(p=2.0)
    with pytest.raises(ValueError, match="1 < p < 2"):
        Tweedie(p=0.5)


def test_tweedie_validmu_and_initialize():
    f = Tweedie(p=1.5)
    assert f.validmu(np.array([0.1, 1.0, 5.0]))
    assert not f.validmu(np.array([0.0, 1.0]))
    assert not f.validmu(np.array([-0.1, 1.0]))
    y = np.array([0.0, 1.0, 2.5])
    np.testing.assert_allclose(f.initialize(y, np.ones(3)), y + 0.1)
    with pytest.raises(ValueError, match="negative values"):
        f.initialize(np.array([-1.0, 1.0]), np.ones(2))


_TW_MUS = np.array([0.5, 1.0, 2.0, 5.0])


@pytest.mark.parametrize(
    "p, V, dV, d2V, d3V",
    [
        (
            1.1,
            [0.466516495768403705, 1.0, 2.143546925072586262, 5.873094715440095648],
            [1.02633629069048826, 1.10000000000000009, 1.17895080878992253, 1.29208083739682111],
            [0.2052672581380978467, 0.1100000000000001116, 0.0589475404394961822, 0.0258416167479364467],
            [-0.3694810646485760519, -0.0990000000000000879, -0.0265263931977732792, -0.0046514910146285603],
        ),
        (
            1.5,
            [0.353553390593273786, 1.0, 2.828427124746190291, 11.180339887498949025],
            [1.06066017177982141, 1.50000000000000000, 2.12132034355964283, 3.35410196624968471],
            [1.060660171779821415, 0.750000000000000000, 0.530330085889910707, 0.335410196624968460],
            [-1.0606601717798214146, -0.3750000000000000000, -0.1325825214724776768, -0.0335410196624968474],
        ),
        (
            1.9,
            [0.267943365634073283, 1.0, 3.732131966147229640, 21.283498063019610669],
            [1.01818478940947843, 1.89999999999999991, 3.54552536783986794, 8.08772926394745184],
            [1.83273262093706091, 1.70999999999999974, 1.59548641552794046, 1.45579126751054133],
            [-0.3665465241874125146, -0.1710000000000001241, -0.0797743207763970952, -0.0291158253502108513],
        ),
    ],
)
def test_tweedie_variance_oracle(p, V, dV, d2V, d3V):
    """mgcv:::fix.family.var(Tweedie(p, link='log'))$dvar/d2var/d3var at mu=(0.5,1,2,5)."""
    f = Tweedie(p=p)
    np.testing.assert_allclose(f.variance(_TW_MUS), V, rtol=1e-13)
    np.testing.assert_allclose(f.dvar(_TW_MUS), dV, rtol=1e-13)
    np.testing.assert_allclose(f.d2var(_TW_MUS), d2V, rtol=1e-13)
    np.testing.assert_allclose(f.d3var(_TW_MUS), d3V, rtol=1e-13)


_TW_DEV_Y = np.array([0.0, 0.5, 1.0, 2.5, 4.0])
_TW_DEV_MU = np.array([0.6, 0.7, 1.2, 2.0, 3.5])


@pytest.mark.parametrize(
    "p, dev_oracle",
    [
        (
            1.1,
            [1.4032130388652341857, 0.0665577307825964692,
             0.0349267878477985683, 0.1071552861439123427, 0.0599447148388876361],
        ),
        (
            1.5,
            [3.0983866769659336171, 0.0802430753127092444,
             0.0332641767424362023, 0.0788114206843384402, 0.0356745147454633482],
        ),
        (
            1.9,
            [19.0040043301135099796, 0.0968397017685943551,
             0.0316900713608997964, 0.0579905047662896411, 0.0212341107387551409],
        ),
    ],
)
def test_tweedie_dev_resids_oracle(p, dev_oracle):
    """Tweedie(p)$dev.resids(y, mu, wt=1)."""
    f = Tweedie(p=p)
    np.testing.assert_allclose(
        f.dev_resids(_TW_DEV_Y, _TW_DEV_MU, np.ones(5)),
        dev_oracle, rtol=1e-13,
    )


def test_tweedie_dev_resids_weighted_oracle():
    """Tweedie(p=1.5)$dev.resids with non-unit prior weights."""
    f = Tweedie(p=1.5)
    wt = np.array([0.5, 1.0, 2.0, 1.0, 0.5])
    oracle = [1.5491933384829668086, 0.0802430753127092444,
              0.0665283534848724045, 0.0788114206843384402, 0.0178372573727316741]
    np.testing.assert_allclose(
        f.dev_resids(_TW_DEV_Y, _TW_DEV_MU, wt), oracle, rtol=1e-13,
    )


def test_tweedie_dev_resids_zero_at_y_equals_mu():
    f = Tweedie(p=1.5)
    y = np.array([0.5, 1.0, 2.0, 5.0])
    np.testing.assert_allclose(f.dev_resids(y, y, np.ones(4)), 0.0, atol=1e-13)


def test_tweedie_dev_limit_to_poisson():
    """Tweedie deviance → Poisson deviance as p → 1."""
    y = np.array([1.0, 2.0, 5.0, 10.0])
    mu = np.array([1.5, 1.5, 4.0, 8.0])
    pois = Poisson().dev_resids(y, mu, np.ones(4))
    np.testing.assert_allclose(
        Tweedie(p=1.001).dev_resids(y, mu, np.ones(4)), pois, rtol=1e-2,
    )


def test_tweedie_dev_limit_to_gamma():
    """Tweedie deviance → Gamma deviance as p → 2."""
    y = np.array([0.5, 1.0, 2.0, 5.0])
    mu = np.array([0.6, 1.5, 2.5, 4.0])
    gam_dev = Gamma().dev_resids(y, mu, np.ones(4))
    np.testing.assert_allclose(
        Tweedie(p=1.999).dev_resids(y, mu, np.ones(4)), gam_dev, rtol=1e-2,
    )


def test_tweedie_log_density_oracle_p15():
    """ldTweedie(y, mu, rho=0, theta=0, a=1.01, b=1.99)[, 1] at p=1.5, phi=1."""
    f = Tweedie(p=1.5)
    log_f = f._log_density(_TW_DEV_Y, _TW_DEV_MU, phi=1.0, wt=np.ones(5))
    oracle = [-1.549193338482966809, -0.608940759999017533,
              -1.045247308713201040, -1.710387087424116714,
              -2.026689918613598707]
    np.testing.assert_allclose(log_f, oracle, rtol=1e-12)


def test_tweedie_log_density_oracle_p17():
    """Same at p=1.7, phi=2.0 — exercises the Dunn-Smyth series at off-default p."""
    f = Tweedie(p=1.7)
    log_f = f._log_density(_TW_DEV_Y, _TW_DEV_MU, phi=2.0, wt=np.ones(5))
    oracle = [-1.429862000740157901, -0.970520782376560809,
              -1.491788879150033331, -2.222992852580873091,
              -2.589272021845887117]
    np.testing.assert_allclose(log_f, oracle, rtol=1e-12)


def test_tweedie_ls_saturated_oracle_p15():
    """tw() at default theta=0 ⇒ p=1.5, scale=1; sums of ldTweedie(y, y, ...)[, 1:3]."""
    f = tw()
    assert f.p == pytest.approx(1.5, abs=1e-12)
    ls = f.ls(_TW_DEV_Y, np.ones(5), scale=1.0)
    np.testing.assert_allclose(ls[0], -5.2772684810074608, rtol=1e-12)
    np.testing.assert_allclose(ls[1], -2.4805433077476438, rtol=1e-12)
    np.testing.assert_allclose(ls[2], -0.6812533145450566, rtol=1e-9)


def test_tweedie_ls_saturated_oracle_p17():
    """tw().set_theta(...) → p=1.7, scale=2."""
    f = tw()
    f.set_theta(np.log((1.7 - 1.01) / (1.99 - 1.7)))
    assert f.p == pytest.approx(1.7, abs=1e-12)
    ls = f.ls(_TW_DEV_Y, np.ones(5), scale=2.0)
    np.testing.assert_allclose(ls[0], -7.22064210632427717, rtol=1e-12)
    np.testing.assert_allclose(ls[1], -2.8497276172663244, rtol=1e-12)
    np.testing.assert_allclose(ls[2], -0.837451847912255021, rtol=1e-9)


def test_tweedie_ls_weighted_oracle():
    """Weighted saturated ls; per-obs scale phi_i = phi/wt_i (mgcv tw()$ls)."""
    f = Tweedie(p=1.5)
    wt = np.array([0.5, 1.0, 2.0, 1.0, 0.5])
    ls = f.ls(_TW_DEV_Y, wt, scale=1.5)
    np.testing.assert_allclose(ls[0], -5.85919331444838587, rtol=1e-12)
    np.testing.assert_allclose(ls[1], -3.06707140557613522, rtol=1e-12)


def test_tweedie_ls_zero_weights_dropped():
    """Rows with wt=0 should drop out of ls (mgcv's good-subset convention)."""
    f = Tweedie(p=1.5)
    y = np.array([0.0, 1.0, 2.5])
    wt_drop = np.array([1.0, 0.0, 1.0])
    ls_drop = f.ls(y, wt_drop, 1.0)
    ls_two = f.ls(np.array([0.0, 2.5]), np.array([1.0, 1.0]), 1.0)
    np.testing.assert_allclose(ls_drop, ls_two, rtol=1e-12)


def test_tw_dls_dp_chain_to_theta_oracle_p15():
    """dls/dp · dp/dθ_tw must equal Σ ldTweedie[, 'th'] at default (a=1.01, b=1.99)."""
    f = tw()
    dls_dp = f.dls_dp(_TW_DEV_Y, np.ones(5), scale=1.0)
    np.testing.assert_allclose(dls_dp * f.dp_dtheta(),
                               -0.1740833687231026, rtol=1e-9)


def test_tw_dls_dp_chain_to_theta_oracle_p17():
    f = tw()
    f.set_theta(np.log((1.7 - 1.01) / (1.99 - 1.7)))
    dls_dp = f.dls_dp(_TW_DEV_Y, np.ones(5), scale=2.0)
    np.testing.assert_allclose(dls_dp * f.dp_dtheta(),
                               -0.0302015497694411161, rtol=1e-9)


def test_tw_default_theta_zero_gives_p_15():
    """θ=0 with default (a,b)=(1.01, 1.99) → p = (1.01+1.99)/2 = 1.5."""
    f = tw()
    assert f.theta == 0.0
    assert f.p == pytest.approx(1.50, abs=1e-12)
    assert f.n_theta == 1
    np.testing.assert_array_equal(f.get_theta(), np.array([0.0]))


def test_tw_set_theta_array_or_scalar():
    """set_theta accepts both scalar and length-1 array (Family-base contract)."""
    f = tw()
    f.set_theta(0.5)
    p_scalar = f.p
    f.set_theta(np.array([0.5]))
    np.testing.assert_allclose(f.p, p_scalar, rtol=1e-13)


def test_tw_dp_dtheta_default_a_b():
    f = tw()
    np.testing.assert_allclose(f.dp_dtheta(), 0.245, rtol=1e-13)
    f.set_theta(2.0)
    s = 1.0 / (1.0 + np.exp(-2.0))
    np.testing.assert_allclose(f.dp_dtheta(), 0.98 * s * (1.0 - s), rtol=1e-13)


def test_tw_invalid_a_b_raises():
    with pytest.raises(ValueError, match="1 ≤ a < b ≤ 2"):
        tw(a=2.0, b=1.5)
    with pytest.raises(ValueError, match="1 ≤ a < b ≤ 2"):
        tw(a=0.9, b=1.5)
    with pytest.raises(ValueError, match="1 ≤ a < b ≤ 2"):
        tw(a=1.5, b=2.5)
