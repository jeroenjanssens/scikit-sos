"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def simple_data() -> np.ndarray:
    """Simple 2D dataset with clear outlier."""
    return np.array(
        [
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [10, 10],  # Clear outlier
        ]
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator with fixed seed."""
    return np.random.default_rng(42)
