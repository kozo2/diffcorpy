"""A focused port of the ``fdrtool`` (Strimmer 2008) local FDR estimator.

Only the ``statistic = "pvalue"`` path used by :func:`diffcorr.get_lfdr` is
reproduced here, following the algorithm in the R ``fdrtool`` package:
cutoff selection (``fndr``), null-model / ``eta0`` estimation, the modified
empirical CDF, and the Grenander density estimator.

Reference: Strimmer, K. (2008) Bioinformatics 24, 1461-1462.
"""
from __future__ import annotations

import numpy as np

__all__ = ["fdrtool_pvalue"]


def _quantile_type7(x: np.ndarray, prob: float) -> float:
    """R's default ``quantile`` (type 7), equivalent to numpy 'linear'."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 1:
        return float(x[0])
    h = (n - 1) * prob
    lo = int(np.floor(h))
    hi = min(lo + 1, n - 1)
    return float(x[lo] + (h - lo) * (x[hi] - x[lo]))


def _isomean(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) least-squares regression (PAVA).

    Mirrors fdrtool's ``C_isomean``; the L2 isotonic fit is unique so this
    pool-adjacent-violators implementation reproduces it exactly.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = y.size
    if n == 1:
        return y.copy()
    val = np.empty(n)
    wt = np.empty(n)
    cnt = np.empty(n, dtype=np.int64)
    b = -1
    for i in range(n):
        b += 1
        val[b] = y[i]
        wt[b] = w[i]
        cnt[b] = 1
        while b > 0 and val[b] < val[b - 1]:
            new_w = wt[b - 1] + wt[b]
            val[b - 1] = (wt[b - 1] * val[b - 1] + wt[b] * val[b]) / new_w
            wt[b - 1] = new_w
            cnt[b - 1] += cnt[b]
            b -= 1
    ghat = np.empty(n)
    idx = 0
    for j in range(b + 1):
        ghat[idx:idx + cnt[j]] = val[j]
        idx += cnt[j]
    return ghat


def _gcmlcm(x: np.ndarray, y: np.ndarray, kind: str):
    """Greatest convex minorant ('gcm') or least concave majorant ('lcm')."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    rawslope = dy / dx
    rawslope[np.isposinf(rawslope)] = np.finfo(float).max
    rawslope[np.isneginf(rawslope)] = -np.finfo(float).max
    if kind == "gcm":
        slope = _isomean(rawslope, dx)
    else:  # lcm
        slope = -_isomean(-rawslope, dx)
    # !duplicated(slope): keep first occurrence of each (consecutive) value
    keep = np.ones(slope.size, dtype=bool)
    keep[1:] = slope[1:] != slope[:-1]
    x_keep = np.concatenate([x[np.concatenate([keep, [True]])]])
    dx_knots = np.diff(x_keep)
    slope_knots = slope[keep]
    y_knots = y[0] + np.concatenate([[0.0], np.cumsum(dx_knots * slope_knots)])
    return x_keep, y_knots, slope_knots


def _pval_estimate_eta0_quantile(p: np.ndarray, q: float = 0.1,
                                 lambdas: np.ndarray | None = None) -> float:
    if lambdas is None:
        lambdas = np.arange(0, 0.95, 0.05)
    p = np.asarray(p, dtype=float)
    e0 = np.array([np.mean(p >= lam) / (1 - lam) for lam in lambdas])
    e0 = np.clip(e0, 0.0, 1.0)
    return _quantile_type7(e0, q)


def _fndr_cutoff_pvalue(x: np.ndarray) -> float:
    ax = 1.0 - x
    e0_guess = _pval_estimate_eta0_quantile(x)  # pval == x for the pvalue case

    def fndrfunc(z: float) -> float:
        f_x = np.mean(ax < z)
        if f_x == 0:
            return 0.0
        return max(0.0, (f_x - e0_guess * z) / f_x)  # F0(z) == z

    MAXPCT0 = 0.99
    zeta0 = _quantile_type7(ax, min(MAXPCT0, e0_guess))
    fndr2 = fndrfunc(zeta0)
    fndr1 = fndrfunc(0.9 * zeta0)
    while fndr1 < fndr2:
        zeta0 = zeta0 * 0.9
        fndr2 = fndr1
        fndr1 = fndrfunc(0.9 * zeta0)
    return 1.0 - zeta0


def _censored_fit_eta0_pvalue(x: np.ndarray, x0: float) -> float:
    n = x.size
    n_cens = int(np.sum(x >= x0))
    m = 1.0 - x0  # 1 - get.pval(x0) for the pvalue null model
    th = n_cens / n
    return min(1.0, th / m)


def _ecdf_pval(x: np.ndarray, eta0: float):
    """Modified empirical CDF on p-values (fdrtool ``ecdf.pval``)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    vals, counts = np.unique(x, return_counts=True)
    f_raw = np.cumsum(counts) / n
    f_raw = np.minimum(f_raw, 1.0 - eta0 * (1.0 - vals))
    f_raw = np.maximum(f_raw, eta0 * vals)
    vals = vals.astype(float)
    if vals[-1] != 1.0:
        f_raw = np.append(f_raw, 1.0)
        vals = np.append(vals, 1.0)
    if vals[0] != 0.0:
        f_raw = np.insert(f_raw, 0, 0.0)
        vals = np.insert(vals, 0, 0.0)
    i = vals.size - 2  # R's length(vals) - 1 (1-based) -> second to last
    f_raw[i] = 1.0 - eta0 * (1.0 - vals[i])
    return vals, f_raw


def _approx_constant(xk: np.ndarray, yk: np.ndarray, q: np.ndarray) -> np.ndarray:
    """approxfun(method='constant', rule=2): left value of the bracketing knot."""
    q = np.asarray(q, dtype=float)
    idx = np.searchsorted(xk, q, side="right") - 1
    idx = np.clip(idx, 0, xk.size - 1)
    return yk[idx]


def fdrtool_pvalue(p, cutoff_method: str = "fndr", pct0: float = 0.75) -> dict:
    """Local FDR for a vector of p-values (``fdrtool(p, statistic='pvalue')``).

    Returns a dict with keys ``pval``, ``qval``, ``lfdr``, ``eta0`` and
    ``param`` (the ``[cutoff, N.cens, eta0, eta0.SE]`` null-model row).
    """
    p = np.asarray(p, dtype=float)
    if p.min() < 0 or p.max() > 1:
        raise ValueError("input p-values must all be in the range 0 to 1!")

    if cutoff_method == "pct0":
        x0 = _quantile_type7(p, 1 - pct0)
    elif cutoff_method == "fndr":
        x0 = _fndr_cutoff_pvalue(p)
    else:
        raise ValueError("cutoff_method must be 'fndr' or 'pct0' for p-values")

    n_cens = int(np.sum(p >= x0))
    eta0 = _censored_fit_eta0_pvalue(p, x0)
    m = 1.0 - x0
    eta0_se = float(np.sqrt((n_cens / p.size) * (1 - n_cens / p.size) /
                            (p.size * m * m)))

    vals, f_raw = _ecdf_pval(p, eta0)
    x_knots, F_knots, slope_knots = _gcmlcm(vals, f_raw, "lcm")
    f_knots = np.append(slope_knots, slope_knots[-1])

    # local fdr: density-based
    f_pval = _approx_constant(x_knots, f_knots, p)
    lfdr = np.minimum(eta0 / f_pval, 1.0)

    # tail-area Fdr -> q-values (linear interpolation of the CDF)
    F_pval = np.interp(p, x_knots, F_knots, left=0.0, right=F_knots[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        qval = np.minimum(np.where(F_pval > 0, eta0 * p / F_pval, 1.0), 1.0)

    return {
        "pval": p,
        "qval": qval,
        "lfdr": lfdr,
        "eta0": eta0,
        "param": np.array([[x0, n_cens, eta0, eta0_se]]),
    }
