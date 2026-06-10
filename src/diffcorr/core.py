"""Core DiffCorr functionality, ported from the R package.

Data convention (same as the R package): a 2-D table with **rows = molecules**
(genes/metabolites) and **columns = samples/replicates**. Inputs may be
:class:`pandas.DataFrame` or :class:`numpy.ndarray`; molecule labels are taken
from the DataFrame index when available.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm, rankdata, t as student_t

from ._fdrtool import fdrtool_pvalue

ArrayLike = Union[pd.DataFrame, np.ndarray, Sequence]

__all__ = [
    "scaling_methods", "cor2_test", "compcorr", "get_lfdr", "comp_2_cc_fdr",
    "cor_dist", "uncent_cor2dist", "uncent_cordist", "cluster_molecule",
    "cutree", "get_eigen_molecule", "generate_g", "get_eigen_molecule_graph",
    "write_modules", "get_min_max",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_matrix(data: ArrayLike):
    """Return (ndarray, row_labels) with rows = molecules."""
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=float), list(data.index.astype(str))
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr, [str(i) for i in range(arr.shape[0])]


def _cor_rows(mat: np.ndarray, method: str = "pearson") -> np.ndarray:
    """Correlation matrix between the rows of ``mat`` (R's ``cor(t(data))``)."""
    method = method.lower()
    A = np.asarray(mat, dtype=float)
    if A.shape[0] == 1:
        return np.ones((1, 1))
    if method == "pearson":
        return np.corrcoef(A)
    if method == "spearman":
        R = np.vstack([rankdata(row) for row in A])
        return np.corrcoef(R)
    if method == "kendall":
        from scipy.stats import kendalltau
        m = A.shape[0]
        C = np.ones((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                C[i, j] = C[j, i] = kendalltau(A[i], A[j])[0]
        return C
    raise ValueError("method must be 'pearson', 'spearman' or 'kendall'")


def _lower_tri_colmajor(n: int):
    """Indices of the strict lower triangle in R's column-major order."""
    rows, cols = [], []
    for j in range(n):
        for i in range(j + 1, n):
            rows.append(i)
            cols.append(j)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def p_adjust(p: np.ndarray, method: str = "BH") -> np.ndarray:
    """Subset of R's ``p.adjust`` (BH/fdr, bonferroni, holm, none)."""
    p = np.asarray(p, dtype=float)
    n = p.size
    method = method.lower()
    if method in ("bh", "fdr"):
        order = np.argsort(p)[::-1]
        ranks = np.arange(n, 0, -1)
        adj = np.minimum.accumulate((n / ranks) * p[order])
        out = np.empty(n)
        out[order] = np.minimum(adj, 1.0)
        return out
    if method == "bonferroni":
        return np.minimum(p * n, 1.0)
    if method == "holm":
        order = np.argsort(p)
        adj = np.maximum.accumulate((n - np.arange(n)) * p[order])
        out = np.empty(n)
        out[order] = np.minimum(adj, 1.0)
        return out
    if method == "none":
        return p
    raise ValueError(f"unsupported p.adjust method: {method}")


# --------------------------------------------------------------------------- #
# scaling
# --------------------------------------------------------------------------- #
def scaling_methods(data: ArrayLike, methods: str = "auto") -> Optional[pd.DataFrame]:
    """Row-wise pre-treatment/scaling of a molecule x sample table.

    ``methods`` is one of ``auto``, ``range``, ``pareto``, ``vast``,
    ``level`` or ``power``. Returns ``None`` when ``data`` has a single column.
    """
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(np.asarray(data, dtype=float))
    if df.shape[1] <= 1:
        return None
    X = df.to_numpy(dtype=float)
    mean = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, ddof=1, keepdims=True)

    if methods == "auto":
        res = (X - mean) / sd
    elif methods == "range":
        rng = (np.nanmax(X, axis=1, keepdims=True) - np.nanmin(X, axis=1, keepdims=True))
        res = (X - mean) / rng
    elif methods == "pareto":
        res = (X - mean) / np.sqrt(sd)
    elif methods == "vast":
        res = mean * (X - mean) / (sd ** 2)
    elif methods == "level":
        res = (X - mean) / mean
    elif methods == "power":
        sq = np.sqrt(X)
        res = sq - np.mean(sq, axis=1, keepdims=True)
    else:
        raise ValueError("methods must be one of auto/range/pareto/vast/level/power")
    return pd.DataFrame(res, index=df.index, columns=df.columns)


# --------------------------------------------------------------------------- #
# correlation tests
# --------------------------------------------------------------------------- #
def cor2_test(n: int, r, method: str = "pearson"):
    """Two-sided p-value for a correlation coefficient ``r`` from ``n`` samples."""
    r = np.asarray(r, dtype=float)
    if method in ("pearson", "spearman"):
        t = np.abs(r) * np.sqrt((n - 2) / (1 - r ** 2))
        p = student_t.sf(t, n - 2) * 2
    elif method == "kendall":
        z = np.abs(r) / np.sqrt((4 * n + 10) / (9 * n * (n - 1)))
        p = norm.sf(z) * 2
    else:
        raise ValueError("method must be 'pearson', 'spearman' or 'kendall'")
    return p


def compcorr(n1: int, r1, n2: int, r2) -> dict:
    """Compare two correlation coefficients with Fisher's Z-transformation.

    Returns ``{"diff": z-score, "pval": two-sided p-value}`` (arrays if the
    correlation inputs are arrays).
    """
    r1 = np.array(r1, dtype=float, ndmin=1).copy()
    r2 = np.array(r2, dtype=float, ndmin=1).copy()
    r1[r1 >= 0.99] = 0.99
    r2[r2 >= 0.99] = 0.99
    r1[r1 <= -0.99] = -0.99
    r2[r2 <= -0.99] = -0.99
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    dz = (z1 - z2) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    pv = 2 * (1 - norm.cdf(np.abs(dz)))
    if dz.size == 1:
        return {"diff": float(dz[0]), "pval": float(pv[0])}
    return {"diff": dz, "pval": pv}


def get_lfdr(r, cutoff_method: str = "fndr") -> dict:
    """Local false discovery rate of a vector of p-values (fdrtool port)."""
    return fdrtool_pvalue(r, cutoff_method=cutoff_method)


# --------------------------------------------------------------------------- #
# differential correlation
# --------------------------------------------------------------------------- #
def comp_2_cc_fdr(data1: ArrayLike, data2: ArrayLike, output_file: str = "res.txt",
                  method: str = "pearson", p_adjust_methods: str = "local",
                  threshold: float = 0.05, save: bool = False) -> pd.DataFrame:
    """Differential correlation between two conditions (Fisher's Z + lFDR).

    ``data1`` / ``data2`` are molecule x sample tables sharing the same
    molecules (rows). Returns a :class:`pandas.DataFrame` of the molecule pairs
    whose differential-correlation significance is below ``threshold``.
    """
    m1, names = _as_matrix(data1)
    m2, _ = _as_matrix(data2)
    cc1 = _cor_rows(m1, method)
    cc2 = _cor_rows(m2, method)
    n = m1.shape[0]
    rows, cols = _lower_tri_colmajor(n)
    ccc1 = cc1[rows, cols]
    ccc2 = cc2[rows, cols]
    n1, n2 = m1.shape[1], m2.shape[1]

    p1 = cor2_test(n1, ccc1, method)
    p2 = cor2_test(n2, ccc2, method)
    pdiff = compcorr(n1, ccc1, n2, ccc2)["pval"]
    pdiff = np.asarray(pdiff, dtype=float)
    diff = ccc1 - ccc2
    pdiff[np.isnan(pdiff)] = 1.0

    pa = p_adjust_methods.lower()
    if pa == "local":
        p1_lfdr = get_lfdr(p1)["lfdr"]
        p2_lfdr = get_lfdr(p2)["lfdr"]
        pdiff_lfdr = get_lfdr(pdiff)["lfdr"]
    elif pa in ("bh", "fdr"):
        p1_lfdr = p_adjust(p1, "BH")
        p2_lfdr = p_adjust(p2, "BH")
        pdiff_lfdr = p_adjust(pdiff, "BH")
    else:
        p1_lfdr = np.array(["not adjusted"] * ccc1.size, dtype=object)
        p2_lfdr = np.array(["not adjusted"] * ccc1.size, dtype=object)
        pdiff_lfdr = np.array(["not adjusted"] * ccc1.size, dtype=object)

    mol_x = [names[j] for j in cols]   # column molecule
    mol_y = [names[i] for i in rows]   # row molecule

    if pa in ("local", "bh", "fdr"):
        keep = np.asarray(pdiff_lfdr, dtype=float) < threshold
    else:
        keep = np.ones(ccc1.size, dtype=bool)

    res = pd.DataFrame({
        "molecule X": np.array(mol_x)[keep],
        "molecule Y": np.array(mol_y)[keep],
        "r1": ccc1[keep],
        "p1": p1[keep],
        "r2": ccc2[keep],
        "p2": p2[keep],
        "p (difference)": pdiff[keep],
        "(r1-r2)": diff[keep],
        "lfdr (in cond. 1)": np.asarray(p1_lfdr)[keep],
        "lfdr (in cond. 2)": np.asarray(p2_lfdr)[keep],
        "lfdr (difference)": np.asarray(pdiff_lfdr)[keep],
    })
    if save:
        res.to_csv(output_file, sep="\t", header=False, index=False)
    return res


# --------------------------------------------------------------------------- #
# distances
# --------------------------------------------------------------------------- #
def cor_dist(data: ArrayLike, methods: str = "pearson", absolute: bool = False) -> np.ndarray:
    """Correlation distance (``1 - r`` or ``1 - |r|``) between molecules (rows)."""
    mat, _ = _as_matrix(data)
    c = _cor_rows(mat, methods)
    return 1 - np.abs(c) if absolute else 1 - c


def uncent_cor2dist(data: ArrayLike, i: int, absolute: bool = False) -> np.ndarray:
    """Uncentered correlation distance of row ``i`` (0-based) to all rows."""
    mat, _ = _as_matrix(data)
    data1_full = mat[i]
    out = np.empty(mat.shape[0])
    for j in range(mat.shape[0]):
        y = mat[j].copy()
        d1 = data1_full.copy()
        mask = ~(np.isnan(y) | np.isnan(d1))
        yy = y[mask]
        dd = d1[mask]
        denom = np.sqrt(np.nansum(yy * yy) * np.nansum(dd * dd))
        out[j] = np.nansum(yy * dd / denom)
    return 1 - np.abs(out) if absolute else 1 - out


def uncent_cordist(data: ArrayLike, absolute: bool = False) -> np.ndarray:
    """Pairwise uncentered correlation distance matrix between molecules (rows).

    Note: the original R function allocated a non-square ``nrow x ncol`` matrix
    (a latent bug, leaving most columns at zero). This port returns the intended
    square ``nrow x nrow`` distance matrix; the populated entries are identical.
    """
    mat, _ = _as_matrix(data)
    m = mat.shape[0]
    out = np.zeros((m, m))
    for i in range(m):
        out[:, i] = uncent_cor2dist(mat, i, absolute)
    return out


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
_LINKAGE_MAP = {
    "average": "average", "single": "single", "complete": "complete",
    "mcquitty": "weighted", "median": "median", "centroid": "centroid",
    "ward": "ward", "ward.d": "ward", "ward.d2": "ward",
}


class HClust:
    """Lightweight analogue of R's ``hclust`` result."""

    def __init__(self, Z: np.ndarray, labels, method: str):
        self.linkage = Z
        self.labels = list(labels)
        self.method = method
        self.height = Z[:, 2].copy()
        from scipy.cluster.hierarchy import leaves_list
        self.order = (leaves_list(Z) + 1).tolist()  # 1-based, like R

    def __repr__(self):
        return f"HClust(n={len(self.labels)}, method={self.method!r}, merges={len(self.height)})"


def cluster_molecule(data: ArrayLike, method: str = "pearson",
                     linkage_method: str = "average", absolute: bool = False) -> HClust:
    """Hierarchical clustering of molecules (rows).

    ``method`` selects the dissimilarity: a correlation measure
    (``pearson``/``spearman``/``kendall``), ``uncentered``, or a
    :func:`scipy.spatial.distance.pdist` metric
    (``euclidean``/``maximum``/``manhattan``/``canberra``/``minkowski``).
    ``linkage_method`` is the agglomeration rule (R names accepted).
    """
    mat, labels = _as_matrix(data)
    if method in ("pearson", "spearman", "kendall"):
        d = squareform(cor_dist(mat, methods=method, absolute=absolute), checks=False)
    elif method == "uncentered":
        d = squareform(uncent_cordist(mat), checks=False)
    elif method in ("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski"):
        metric = {"maximum": "chebyshev", "manhattan": "cityblock"}.get(method, method)
        d = pdist(mat, metric=metric)
    else:
        raise ValueError("method must be a correlation, distance metric or 'uncentered'")

    lk = linkage_method.lower()
    if lk not in _LINKAGE_MAP:
        raise ValueError(f"unknown linkage: {linkage_method}")
    if lk.startswith("ward"):
        warnings.warn("R's 'ward' (ward.D) has no exact SciPy equivalent; "
                      "using SciPy 'ward' (ward.D2).", stacklevel=2)
    Z = linkage(d, method=_LINKAGE_MAP[lk])
    return HClust(Z, labels, _LINKAGE_MAP[lk])


def cutree(hc: HClust, h: Optional[float] = None, k: Optional[int] = None) -> pd.Series:
    """Cut a dendrogram into flat groups (R-compatible group numbering).

    Provide either a height ``h`` or a number of clusters ``k``. Group ids are
    relabelled in order of first appearance across observations, matching R.
    """
    if h is not None:
        raw = fcluster(hc.linkage, t=h, criterion="distance")
    elif k is not None:
        raw = fcluster(hc.linkage, t=k, criterion="maxclust")
    else:
        raise ValueError("supply either h or k")
    remap, nxt, out = {}, 1, np.empty(raw.size, dtype=int)
    for idx, c in enumerate(raw):
        if c not in remap:
            remap[c] = nxt
            nxt += 1
        out[idx] = remap[c]
    return pd.Series(out, index=hc.labels)


# --------------------------------------------------------------------------- #
# eigen molecules
# --------------------------------------------------------------------------- #
def _pca_loadings(submat: np.ndarray, n_pcs: int) -> np.ndarray:
    """First-PC loadings via centred SVD (pcaMethods 'svd' equivalent)."""
    X = submat.T.astype(float)            # samples x molecules
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt.T                           # molecules x components


def get_eigen_molecule(data: ArrayLike, groups, whichgroups=None,
                       methods: str = "svd", n: int = 10) -> dict:
    """First-PC 'eigen-molecule' summary for each sufficiently large group.

    Groups smaller than ``n`` are skipped by default. Returns a dict with
    ``group``, ``N``, ``mean_corr`` and ``eigen_molecules`` (first ``n`` PC1
    loadings per group).
    """
    mat, _ = _as_matrix(data)
    groups = np.asarray(pd.Series(groups).to_numpy())
    uniq = pd.unique(groups)
    sizes = {g: int(np.sum(groups == g)) for g in uniq}

    if whichgroups is None:
        whichgroups = [g for g in uniq if sizes[g] >= n]

    group, N, mean_corr, eigen_molecules = [], [], [], []
    for gi in whichgroups:
        submat = mat[groups == gi]
        c1 = np.corrcoef(submat)
        loadings = _pca_loadings(submat, n)
        eigen_molecules.append(loadings[:n, 0])
        group.append(gi if not isinstance(gi, np.generic) else gi.item())
        N.append(submat.shape[0])
        tri = c1[np.tril_indices(c1.shape[0], -1)]
        mean_corr.append(float(np.mean(tri)))
    return {"group": group, "N": N, "mean_corr": mean_corr,
            "eigen_molecules": eigen_molecules}


# --------------------------------------------------------------------------- #
# graphs
# --------------------------------------------------------------------------- #
def generate_g(data: ArrayLike, method: str = "pearson", cor_thr: float = 0.6,
               neg_flag: int = 1, node_col: str = "red", node_size: int = 7,
               edge_col: str = "blue", edge_width: int = 3):
    """Build an undirected correlation network (NetworkX ``Graph``).

    An edge connects two molecules (rows) whose correlation exceeds ``cor_thr``
    (both signs when ``neg_flag == 1``, positive only otherwise).
    """
    import networkx as nx
    mat, labels = _as_matrix(data)
    c = _cor_rows(mat, method)
    if neg_flag == 1:
        adj = (c >= cor_thr) | (c < -cor_thr)
    else:
        adj = c >= cor_thr

    g = nx.Graph()
    for idx, name in enumerate(labels):
        g.add_node(idx, name=name, label=name, size=node_size, color=node_col)
    m = c.shape[0]
    for i in range(m):
        for j in range(i + 1, m):
            if adj[i, j]:
                g.add_edge(i, j, color=edge_col, width=edge_width)
    return g


def get_eigen_molecule_graph(eigen_list: dict, label: str = "Module"):
    """Correlation network between eigen-molecule modules from :func:`get_eigen_molecule`."""
    ems = eigen_list["eigen_molecules"]
    mat = np.vstack(ems)
    names = [f"{label}{i + 1}" for i in range(len(ems))]
    df = pd.DataFrame(mat, index=names)
    return generate_g(df)


def write_modules(cutree_res: pd.Series, mod_list: dict,
                  outfile: str = "module_list.txt") -> None:
    """Write the molecules of each module to a tab-separated text file."""
    cutree_res = pd.Series(cutree_res)
    with open(outfile, "w") as fh:
        fh.write("Molecule number\tFeature name (probeset or metabolite)\n")
        for gi in mod_list["group"]:
            members = cutree_res[cutree_res == gi].index
            for name in members:
                fh.write(f"{gi}\t{name}\n")


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def get_min_max(d: ArrayLike) -> dict:
    """Maximum and minimum of a table, plus the (1-based) row of the maximum."""
    arr = d.to_numpy(dtype=float) if isinstance(d, pd.DataFrame) else np.asarray(d, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    nrow = arr.shape[0]
    flat_f = arr.flatten(order="F")
    max_lin = int(np.argmax(flat_f))      # 0-based column-major
    min_lin = int(np.argmin(flat_f))
    max_row = max_lin % nrow
    max_col = max_lin // nrow
    min_row = min_lin % nrow
    min_col = min_lin // nrow
    return {"max": float(arr[max_row, max_col]),
            "min": float(arr[min_row, min_col]),
            "max_i": max_row + 1}
