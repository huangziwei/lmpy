"""Distribution of a linear combination of chi-squared variables.

Direct port of mgcv's ``psum.chisq`` (Simon N. Wood, Feb 2020):
- Davies, R. B. (1980) "The Distribution of a Linear Combination of chi^2
  Random Variables" Algorithm AS 155, JRSS-C 29, 323-333.
- Liu, H., Tang, Y., Zhang, H. H. (2009) "A new chi-square approximation to
  the distribution of non-negative definite quadratic forms in non-central
  normal variables" CSDA 53, 853-856 — used as a fallback when Davies fails.

mgcv exposes this as the R-callable ``psum.chisq``; it is the engine behind
``reTest`` (Wood 2013) and the fractional-rank correction inside ``testStat``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2 as _chi2


def _ln1(x: float, first: bool) -> float:
    return math.log1p(x) if first else (math.log1p(x) - x)


def _errbd(u: float, sigsq: float, n: np.ndarray, lb: np.ndarray,
           nc: np.ndarray) -> tuple[float, float]:
    """mgcv davies::errbd — bound on tail probability, returns (errbd, cx)."""
    cx = u * sigsq
    sum1 = u * cx
    u2 = u * 2.0
    r = lb.size
    for j in range(r - 1, -1, -1):
        nj = n[j]
        lj = lb[j]
        ncj = nc[j]
        x = u2 * lj
        y = 1.0 - x
        cx += lj * (ncj / y + nj) / y
        xy = x / y
        sum1 += ncj * xy * xy + nj * (x * xy + _ln1(-x, False))
    return math.exp(-0.5 * sum1), cx


def _ctff(accx: float, upn: float, mean: float, lmin: float, lmax: float,
          sigsq: float, n: np.ndarray, lb: np.ndarray,
          nc: np.ndarray) -> tuple[float, float]:
    """mgcv davies::ctff — find ctff so Pr(qf>ctff)<accx (upn>0) or
    Pr(qf<ctff)<accx (upn<0). Returns (cutoff, upn_out)."""
    u2 = upn
    u1 = 0.0
    c1 = mean
    rb = 2.0 * lmax if u2 > 0 else 2.0 * lmin
    while True:
        eb, c2 = _errbd(u2 / (1.0 + u2 * rb), sigsq, n, lb, nc)
        if eb <= accx:
            break
        u1 = u2
        c1 = c2
        u2 *= 2.0
    while True:
        denom = c2 - mean
        if denom == 0.0:
            break
        if (c1 - mean) / denom >= 0.9:
            break
        u = (u1 + u2) * 0.5
        eb, cst = _errbd(u / (1.0 + u * rb), sigsq, n, lb, nc)
        if eb > accx:
            u1 = u
            c1 = cst
        else:
            u2 = u
            c2 = cst
    return c2, u2


def _truncation(u: float, tausq: float, sigsq: float, n: np.ndarray,
                lb: np.ndarray, nc: np.ndarray) -> float:
    """mgcv davies::truncation — bound integration error from cutoff."""
    pi = math.pi
    sum1 = 0.0
    prod2 = 0.0
    prod3 = 0.0
    s = 0
    sum2 = (sigsq + tausq) * u * u
    prod1 = 2.0 * sum2
    u = u * 2.0
    r = lb.size
    for j in range(r):
        lj = lb[j]
        ncj = nc[j]
        nj = n[j]
        x = u * lj
        x = x * x
        sum1 += ncj * x / (1.0 + x)
        if x > 1.0:
            prod2 += nj * math.log(x)
            prod3 += nj * _ln1(x, True)
            s += nj
        else:
            prod1 += nj * _ln1(x, True)
    sum1 *= 0.5
    prod2 += prod1
    prod3 += prod1
    x = math.exp(-sum1 - 0.25 * prod2) / pi
    y = math.exp(-sum1 - 0.25 * prod3) / pi
    err1 = 1.0 if s == 0 else 2.0 * x / s
    err2 = 2.5 * y if prod3 > 1.0 else 1.0
    if err2 < err1:
        err1 = err2
    x = 0.5 * sum2
    err2 = 1.0 if x <= y else y / x
    return err1 if err1 < err2 else err2


def _findu(utx: float, accx: float, sigsq: float, n: np.ndarray,
           lb: np.ndarray, nc: np.ndarray) -> float:
    """mgcv davies::findu — locate u such that truncation(u) ~ accx."""
    a = (2.0, 1.4, 1.2, 1.1)
    ut = utx
    u = ut * 0.25
    if _truncation(u, 0.0, sigsq, n, lb, nc) > accx:
        while _truncation(ut, 0.0, sigsq, n, lb, nc) > accx:
            ut *= 4.0
    else:
        ut = u
        u = u / 4.0
        while _truncation(u, 0.0, sigsq, n, lb, nc) <= accx:
            ut = u
            u = u / 4.0
    for ai in a:
        u = ut / ai
        if _truncation(u, 0.0, sigsq, n, lb, nc) <= accx:
            ut = u
    return ut


def _integrate(nterm: int, interv: float, tausq: float, main: bool,
               c: float, sigsq: float, n: np.ndarray, lb: np.ndarray,
               nc: np.ndarray, intl: float, ersm: float) -> tuple[float, float]:
    """mgcv davies::integrate — running update of integral and error sums."""
    pi = math.pi
    inpi = interv / pi
    r = lb.size
    for k in range(nterm, -1, -1):
        u = (k + 0.5) * interv
        sum1 = -2.0 * u * c
        sum2 = abs(sum1)
        sum3 = -0.5 * sigsq * u * u
        for j in range(r - 1, -1, -1):
            nj = n[j]
            x = 2.0 * lb[j] * u
            y = x * x
            sum3 -= 0.25 * nj * _ln1(y, True)
            y = nc[j] * x / (1.0 + y)
            z = nj * math.atan(x) + y
            sum1 += z
            sum2 += abs(z)
            sum3 += -0.5 * x * y
        x = inpi * math.exp(sum3) / u
        if not main:
            x *= (1.0 - math.exp(-0.5 * tausq * u * u))
        sum1 = math.sin(0.5 * sum1) * x
        sum2 = 0.5 * sum2 * x
        intl += sum1
        ersm += sum2
    return intl, ersm


def _cfe(x: float, th: np.ndarray, ln28: float, n: np.ndarray, lb: np.ndarray,
         nc: np.ndarray) -> tuple[float, bool]:
    """mgcv davies::cfe — coef of tausq in error from convergence factor.

    Returns (coef, fail)."""
    pi = math.pi
    axl = abs(x)
    sxl = -1 if x < 0 else 1
    sum1 = 0.0
    r = lb.size
    for j in range(r - 1, -1, -1):
        t = int(th[j])
        if lb[t] * sxl > 0.0:
            lj = abs(lb[t])
            axl1 = axl - lj * (n[t] + nc[t])
            axl2 = lj / ln28
            if axl1 > axl2:
                axl = axl1
            else:
                if axl > axl2:
                    axl = axl2
                sum1 = (axl - axl1) / lj
                for k in range(j - 1, -1, -1):
                    sum1 += n[int(th[k])] + nc[int(th[k])]
                break
    if sum1 > 100.0:
        return 1.0, True
    return (2.0 ** (sum1 * 0.25)) / (pi * axl * axl), False


def _davies(lb: np.ndarray, nc: np.ndarray, n: np.ndarray, sigma: float,
            c: float, lim: int, acc: float) -> tuple[float, int]:
    """Direct port of mgcv davies(...). Computes Pr(Q < c) where
    Q = sum_j lb[j] X_j + sigma X_0, X_j ~ chi^2_n[j](nc[j]), X_0 ~ N(0,1).

    Returns (cdf, ifault). ifault: 0=ok; 1=accuracy not met;
    2=round-off concern; 3=invalid params; 4=can't locate params.
    """
    ln28 = math.log(2.0) / 8.0
    pi = math.pi
    intl = 0.0
    ersm = 0.0
    acc1 = float(acc)

    r = lb.size
    # Order indices by descending |lb|.
    th = np.argsort(-np.abs(lb)).astype(int)

    sd = sigma * sigma
    sigsq = sd
    lmax = 0.0
    lmin = 0.0
    mean = 0.0
    for j in range(r):
        nj = n[j]
        lj = lb[j]
        ncj = nc[j]
        if nj < 0 or ncj < 0:
            return 0.0, 3
        sd += lj * lj * (2.0 * nj + 4.0 * ncj)
        mean += lj * (nj + ncj)
        if lmax < lj:
            lmax = lj
        elif lmin > lj:
            lmin = lj
    if sd == 0.0:
        return (1.0 if c > 0.0 else 0.0), 0
    if lmin == 0.0 and lmax == 0.0 and sigma == 0.0:
        return 0.0, 3

    sd = math.sqrt(sd)
    almx = -lmin if lmax < -lmin else lmax

    utx = 16.0 / sd
    up = 4.5 / sd
    un = -up

    utx = _findu(utx, 0.5 * acc1, sigsq, n, lb, nc)

    if c != 0.0 and almx > 0.07 * sd:
        cf, fail = _cfe(c, th, ln28, n, lb, nc)
        tausq = 0.25 * acc1 / cf
        if not fail:
            if _truncation(utx, tausq, sigsq, n, lb, nc) < 0.2 * acc1:
                sigsq = sigsq + tausq
                utx = _findu(utx, 0.25 * acc1, sigsq, n, lb, nc)

    acc1 = 0.5 * acc1

    ok = True
    while ok:
        d1, up = _ctff(acc1, up, mean, lmin, lmax, sigsq, n, lb, nc)
        d1 -= c
        if d1 < 0.0:
            return 1.0, 0
        d2_val, un = _ctff(acc1, un, mean, lmin, lmax, sigsq, n, lb, nc)
        d2 = c - d2_val
        if d2 < 0.0:
            return 0.0, 0
        intv = 2.0 * pi / (d1 if d1 > d2 else d2)
        x = utx / intv
        nt = int(math.floor(x))
        if x - nt > 0.5:
            nt += 1
        x = 3.0 / math.sqrt(acc1)
        ntm = int(math.floor(x))
        if x - ntm > 0.5:
            ntm += 1
        if nt > ntm * 1.5:
            intv1 = utx / ntm
            x = 2.0 * pi / intv1
            if x <= abs(c):
                break
            cf1, fail1 = _cfe(c - x, th, ln28, n, lb, nc)
            cf2, fail2 = _cfe(c + x, th, ln28, n, lb, nc)
            tausq = 0.33 * acc1 / (1.1 * (cf1 + cf2))
            if fail1 or fail2:
                break
            acc1 *= 0.67
            if ntm > lim:
                return 0.0, 1
            intl, ersm = _integrate(ntm, intv1, tausq, False, c, sigsq, n, lb,
                                    nc, intl, ersm)
            lim -= ntm
            sigsq = sigsq + tausq
            utx = _findu(utx, 0.25 * acc1, sigsq, n, lb, nc)
            acc1 = 0.75 * acc1
        else:
            ok = False

    if nt > lim:
        return 0.0, 1
    intl, ersm = _integrate(nt, intv, 0.0, True, c, sigsq, n, lb, nc, intl,
                            ersm)
    cdf = 0.5 - intl

    ifault = 0
    x = ersm + acc / 10.0
    j = 1
    for _ in range(4):
        if j * x == j * ersm:
            ifault = 2
        j *= 2
    return cdf, ifault


def _liu2(x: float, lb: np.ndarray, h: np.ndarray) -> float:
    """mgcv:::liu2 (Liu-Tang-Zhang 2009) survival probability fallback."""
    lh = lb * h
    muQ = float(lh.sum())
    lh = lh * lb
    c2 = float(lh.sum())
    lh = lh * lb
    c3 = float(lh.sum())
    if x <= 0.0 or c2 <= 0.0:
        return 1.0
    s1 = c3 / (c2 ** 1.5)
    s2 = float((lh * lb).sum()) / (c2 ** 2)
    sigQ = math.sqrt(2.0 * c2)
    t = (x - muQ) / sigQ
    if s1 * s1 > s2:
        a = 1.0 / (s1 - math.sqrt(s1 * s1 - s2))
        delta = s1 * a ** 3 - a * a
        l = a * a - 2.0 * delta
    else:
        a = 1.0 / s1
        delta = 0.0
        if c3 == 0.0:
            return 1.0
        l = c2 ** 3 / (c3 ** 2)
    muX = l + delta
    sigX = math.sqrt(2.0) * a
    arg = t * sigX + muX
    if delta == 0.0:
        return float(_chi2.sf(arg, df=l))
    from scipy.stats import ncx2
    return float(ncx2.sf(arg, df=l, nc=delta))


def psum_chisq(q: float, lb: np.ndarray, df: np.ndarray | None = None,
               nc: np.ndarray | None = None, sigma: float = 0.0,
               lower_tail: bool = False, tol: float = 2e-5,
               nlim: int = 100_000) -> float:
    """Survival (or CDF) of Q = sum_j lb[j] * chi^2_{df[j]}(nc[j]) + sigma * N(0,1).

    Mirrors mgcv's ``psum.chisq(q, lb, df, nc, sigz=sigma, lower.tail=...)``.
    Returns Pr(Q > q) by default (lower_tail=False).

    Falls back on Liu-Tang-Zhang (2009) when Davies' algorithm fails to
    converge — same fallback strategy as mgcv.
    """
    lb = np.ascontiguousarray(lb, dtype=float).ravel()
    r = lb.size
    if df is None:
        df = np.ones(r, dtype=int)
    else:
        df = np.asarray(df).ravel()
        if df.size == 1 and r > 1:
            df = np.repeat(df, r)
        df = np.rint(df).astype(int)
    if nc is None:
        nc = np.zeros(r, dtype=float)
    else:
        nc = np.asarray(nc, dtype=float).ravel()
        if nc.size == 1 and r > 1:
            nc = np.repeat(nc, r)
    if df.size != r or nc.size != r:
        raise ValueError("lengths of lb, df, nc must match")
    if (df < 1).any():
        raise ValueError("df must be positive integers")
    if (lb == 0).all():
        raise ValueError("at least one element of lb must be non-zero")
    sigma = max(0.0, float(sigma))

    cdf, ifault = _davies(lb, nc, df, sigma, float(q), int(nlim), float(tol))
    if ifault not in (0, 2):
        # Davies failed — fall back to Liu approximation when central.
        if (nc == 0).all():
            sf = _liu2(float(q), lb, df.astype(float))
            return sf if not lower_tail else 1.0 - sf
        return float("nan")
    sf = 1.0 - cdf
    sf = min(1.0, max(0.0, sf))
    return sf if not lower_tail else 1.0 - sf
