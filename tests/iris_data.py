"""Provide R's built-in ``iris`` dataset (exported verbatim) for tests."""
from functools import lru_cache
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent / "ref" / "iris.csv"


@lru_cache(maxsize=1)
def iris_df() -> pd.DataFrame:
    """Numeric+Species iris table matching R's ``datasets::iris``."""
    return pd.read_csv(_CSV)
