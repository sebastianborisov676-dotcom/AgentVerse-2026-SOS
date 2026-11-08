import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def scene():
    from autocharge import build_model

    return build_model()
