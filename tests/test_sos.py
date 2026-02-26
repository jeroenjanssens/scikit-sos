"""Tests for core SOS algorithm."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from sksos import SOS


class TestSOSInitialization:
    """Test SOS initialization."""

    def test_default_parameters(self) -> None:
        detector = SOS()
        assert detector.perplexity == 30
        assert detector.metric == 'euclidean'
        assert detector.eps == 1e-5

    def test_custom_parameters(self) -> None:
        detector = SOS(perplexity=15, metric='manhattan', eps=1e-6)
        assert detector.perplexity == 15
        assert detector.metric == 'manhattan'
        assert detector.eps == 1e-6


class TestSOSTransformations:
    """Test individual transformation methods."""

    def test_x2d_euclidean(self, simple_data: np.ndarray) -> None:
        detector = SOS(metric='euclidean')
        detector.fit(simple_data)
        D = detector._x2d(simple_data)
        assert D.shape == (5, 5)
        assert np.allclose(D.diagonal(), 0)
        assert np.allclose(D, D.T)

    def test_x2d_none_metric(self) -> None:
        D_input = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        detector = SOS(metric='none')
        detector.fit(D_input)
        D = detector._x2d(D_input)
        assert_array_equal(D, D_input)

    def test_x2d_none_metric_invalid_shape(self) -> None:
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        detector = SOS(metric='none')
        with pytest.raises(ValueError, match='square dissimilarity matrix'):
            detector.fit(X)

    def test_d2a_affinity_properties(self, simple_data: np.ndarray) -> None:
        detector = SOS()
        detector.fit(simple_data)
        D = detector._x2d(simple_data)
        A = detector._d2a(D)
        assert A.shape == D.shape
        assert np.all(A >= 0)

    def test_a2b_sum_to_one(self, simple_data: np.ndarray) -> None:
        detector = SOS()
        detector.fit(simple_data)
        D = detector._x2d(simple_data)
        A = detector._d2a(D)
        B = detector._a2b(A)
        row_sums = B.sum(axis=1)
        assert_array_almost_equal(row_sums, np.ones(len(simple_data)))

    def test_b2o_range(self, simple_data: np.ndarray) -> None:
        detector = SOS()
        detector.fit(simple_data)
        D = detector._x2d(simple_data)
        A = detector._d2a(D)
        B = detector._a2b(A)
        O = detector._b2o(B)
        assert np.all(O >= 0)
        assert np.all(O <= 1)


class TestSOSPredict:
    """Test main predict method."""

    def test_predict_output_shape(self, simple_data: np.ndarray) -> None:
        detector = SOS()
        detector.fit(simple_data)
        O = detector.predict(simple_data)
        assert O.shape == (len(simple_data),)

    def test_predict_identifies_outlier(self, simple_data: np.ndarray) -> None:
        detector = SOS(perplexity=2)
        detector.fit(simple_data)
        O = detector.predict(simple_data)
        assert np.argmax(O) == 4

    @pytest.mark.parametrize('perplexity', [5, 10, 20, 30])
    def test_predict_different_perplexities(
        self, simple_data: np.ndarray, perplexity: float
    ) -> None:
        detector = SOS(perplexity=perplexity)
        detector.fit(simple_data)
        O = detector.predict(simple_data)
        assert O.shape == (len(simple_data),)
        assert np.all(O >= 0)
        assert np.all(O <= 1)

    def test_predict_single_point(self) -> None:
        X = np.array([[1, 2]])
        detector = SOS()
        detector.fit(X)
        O = detector.predict(X)
        assert O.shape == (1,)

    def test_predict_identical_points(self) -> None:
        X = np.ones((5, 3))
        detector = SOS()
        detector.fit(X)
        O = detector.predict(X)
        assert np.allclose(O, O[0], rtol=0.1)


class TestSOSMetrics:
    """Test different distance metrics."""

    def test_euclidean_metric(self, simple_data: np.ndarray) -> None:
        detector = SOS(metric='euclidean')
        detector.fit(simple_data)
        O = detector.predict(simple_data)
        assert O.shape == (len(simple_data),)

    @pytest.mark.parametrize('metric', ['cityblock', 'cosine', 'chebyshev'])
    def test_scipy_metrics(self, simple_data: np.ndarray, metric: str) -> None:
        pytest.importorskip('scipy')
        detector = SOS(metric=metric)
        detector.fit(simple_data)
        O = detector.predict(simple_data)
        assert O.shape == (len(simple_data),)


class TestSOSSklearnCompatibility:
    """Test scikit-learn compatibility."""

    def test_fit_returns_self(self, simple_data: np.ndarray) -> None:
        detector = SOS()
        result = detector.fit(simple_data)
        assert result is detector

    def test_fit_predict_chain(self, simple_data: np.ndarray) -> None:
        detector = SOS()
        O = detector.fit(simple_data).predict(simple_data)
        assert O.shape == (len(simple_data),)


class TestSOSEdgeCases:
    """Test edge cases and numerical stability."""

    def test_high_dimensional_data(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((20, 100))
        detector = SOS()
        detector.fit(X)
        O = detector.predict(X)
        assert O.shape == (20,)

    def test_large_dataset(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((500, 10))
        detector = SOS()
        detector.fit(X)
        O = detector.predict(X)
        assert O.shape == (500,)

    def test_very_large_distances(self) -> None:
        X = np.array([[0, 0], [1e6, 1e6]], dtype=float)
        detector = SOS()
        detector.fit(X)
        O = detector.predict(X)
        assert not np.any(np.isnan(O))
        assert not np.any(np.isinf(O))
