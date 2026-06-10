"""Bundled example datasets, mirroring the ``data/`` objects of the R package."""
from __future__ import annotations

from importlib import resources

import pandas as pd

__all__ = ["load_AraMetLeaves", "load_AraMetRoots", "load_golub_df", "load_pvalues"]


def _read(name: str) -> pd.DataFrame:
    with resources.files("diffcorr.data").joinpath(name).open("r") as fh:
        return pd.read_csv(fh, index_col=0)


def load_AraMetLeaves() -> pd.DataFrame:
    """Arabidopsis leaf metabolome (GC-TOF/MS), 59 metabolites x 50 samples.

    Rows are metabolites, columns are samples (Col-0, tt4, mto1).
    Reference: Kusano, Fukushima et al. BMC Syst Biol 2007 1:53.
    """
    return _read("AraMetLeaves.csv")


def load_AraMetRoots() -> pd.DataFrame:
    """Arabidopsis root metabolome (GC-TOF/MS), 59 metabolites x 53 samples.

    Reference: Fukushima et al. BMC Syst Biol 2011 5:1.
    """
    return _read("AraMetRoots.csv")


def load_golub_df() -> pd.DataFrame:
    """Golub et al. (1999) leukaemia microarray data, 2568 genes x 38 samples.

    Reference: Watson M. BMC Bioinformatics 2006 7:509.
    """
    return _read("golub.df.csv")


def load_pvalues() -> pd.Series:
    """The ``pvalues`` example vector from the R ``fdrtool`` package (length 4289)."""
    with resources.files("diffcorr.data").joinpath("pvalues.csv").open("r") as fh:
        return pd.read_csv(fh)["pvalues"]
