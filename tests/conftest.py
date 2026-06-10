import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

REF = Path(__file__).resolve().parent / "ref"


@pytest.fixture
def ref():
    def _load(name):
        return pd.read_csv(REF / name)
    return _load
