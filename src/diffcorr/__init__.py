"""DiffCorr: analyzing and visualizing differential correlation networks.

A Python port of the R package ``DiffCorr``. It identifies pattern changes in
correlation networks between two experimental conditions, building on Pearson's
correlation and Fisher's Z-test, for omics data such as gene co-expression and
metabolomics.
"""
from .core import (
    HClust,
    comp_2_cc_fdr,
    compcorr,
    cor2_test,
    cor_dist,
    cluster_molecule,
    cutree,
    generate_g,
    get_eigen_molecule,
    get_eigen_molecule_graph,
    get_lfdr,
    get_min_max,
    p_adjust,
    scaling_methods,
    uncent_cor2dist,
    uncent_cordist,
    write_modules,
)
from .datasets import (
    load_AraMetLeaves,
    load_AraMetRoots,
    load_golub_df,
    load_pvalues,
)

__version__ = "0.4.5"

__all__ = [
    "HClust", "comp_2_cc_fdr", "compcorr", "cor2_test", "cor_dist",
    "cluster_molecule", "cutree", "generate_g", "get_eigen_molecule",
    "get_eigen_molecule_graph", "get_lfdr", "get_min_max", "p_adjust",
    "scaling_methods", "uncent_cor2dist", "uncent_cordist", "write_modules",
    "load_AraMetLeaves", "load_AraMetRoots", "load_golub_df", "load_pvalues",
    "plot_cluster_molecules", "plot_diff_corr_group",
]


def __getattr__(name):  # lazy import of plotting (optional matplotlib dep)
    if name in ("plot_cluster_molecules", "plot_diff_corr_group"):
        from . import plotting
        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
