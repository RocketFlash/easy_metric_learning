import random
from types import SimpleNamespace

import pytest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@pytest.fixture(autouse=True)
def seed_tests():
    random.seed(28)
    if np is not None:
        np.random.seed(28)
    if torch is not None:
        torch.manual_seed(28)


@pytest.fixture
def dummy_config():
    return SimpleNamespace(debug=False)
