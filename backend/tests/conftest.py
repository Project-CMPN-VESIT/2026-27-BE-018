import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_TMP = tempfile.mkdtemp(prefix="gln-test-")
os.environ["GLN_DATA_DIR"] = _TMP
os.environ.setdefault("GLN_API_KEY", "")
os.environ["GLN_STATIC_DIR"] = str(Path(_TMP) / "no-ui")


@pytest.fixture(scope="session")
def engine():
    from app.config import get_settings
    from app.ml.engine import ModelEngine
    e = ModelEngine(get_settings().model_path, "cpu", 0, 4, 64)
    e.load()
    return e
