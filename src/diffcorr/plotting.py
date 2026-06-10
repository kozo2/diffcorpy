"""Plotting helpers (matplotlib), ported from the R package's base-graphics code."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["plot_cluster_molecules", "plot_diff_corr_group"]


def _to_df(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(np.asarray(data, dtype=float))


def plot_cluster_molecules(data, groups=None, group_no=None, title=None,
                           ylim=None, order=None, scale_center=False,
                           scale_scale=False, frame="white", col=None,
                           xlab="Samples", ylab="Relative abundance", ax=None):
    """Line plot of the abundance profiles of one cluster of molecules.

    Returns the matplotlib ``Axes``. Requires the optional ``matplotlib`` extra.
    """
    import matplotlib.pyplot as plt

    df = _to_df(data)
    if groups is not None and group_no is not None:
        mask = np.asarray(pd.Series(groups).to_numpy()) == group_no
        df = df.loc[df.index[mask]] if df.shape[0] == mask.size else df.iloc[mask]

    X = df.to_numpy(dtype=float)
    if scale_center or scale_scale:
        center = X.mean(axis=1, keepdims=True) if scale_center else 0.0
        scale = X.std(axis=1, ddof=1, keepdims=True) if scale_scale else 1.0
        X = (X - center) / scale

    red = None
    if order is not None:
        if order == "average":
            red = X.mean(axis=0)
            d1 = np.argsort(red)
            red = red[d1]
        else:
            row = list(df.index).index(order)
            d1 = np.argsort(X[row])
            red = X[row, d1]
        X = X[:, d1]

    nrow = X.shape[0]
    if col is None:
        colors = ["black"] * nrow
    elif col == "random":
        cmap = plt.cm.rainbow(np.linspace(0, 1, nrow))
        colors = [tuple(c) for c in cmap]
    elif isinstance(col, str):
        colors = [col] * nrow
    else:
        colors = list(col)

    if ax is None:
        _, ax = plt.subplots()
    if ylim is None:
        ylim = (np.nanmin(X), np.nanmax(X))
    ax.set_facecolor(frame)
    xs = np.arange(1, X.shape[1] + 1)
    for i in range(nrow):
        ax.plot(xs, X[i], color=colors[i])
    if red is not None:
        ax.plot(xs, red, color="red", linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(list(df.columns), rotation=90)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    return ax


def plot_diff_corr_group(data, groups1=None, groups2=None, group1_no=None,
                         group2_no=None, g1=None, g2=None, g1_order=None,
                         g2_order=None, title1=None, title2=None, **kwargs):
    """Side-by-side cluster plots for one module under two conditions.

    ``g1`` / ``g2`` are the column indices (0-based) belonging to each
    condition. Returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    df = _to_df(data)
    mask1 = np.asarray(pd.Series(groups1).to_numpy()) == group1_no
    mask2 = np.asarray(pd.Series(groups2).to_numpy()) == group2_no
    data1 = df.iloc[mask1]
    data2 = df.iloc[mask2]
    d1 = data1.iloc[:, list(g1)]
    d2 = data2.iloc[:, list(g2)]

    fig, axes = plt.subplots(1, 2)
    plot_cluster_molecules(d1, order=g1_order, title=title1, ax=axes[0], **kwargs)
    plot_cluster_molecules(d2, order=g2_order, title=title2, ax=axes[1], **kwargs)
    return fig
