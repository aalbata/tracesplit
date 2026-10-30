from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
@pytest.fixture
def root(): return ROOT
@pytest.fixture
def source(root): return (root/"examples"/"iris.tsml").read_text()
