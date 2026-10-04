import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests" / "fixtures" / "mini_hle.parquet"


@pytest.fixture(scope="session")
def questions():
    from jeopardybench.data import load_questions
    return load_questions(FIXTURE)
