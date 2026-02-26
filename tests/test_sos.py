"""Tests for core SOS algorithm."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from sksos import SOS


class TestSOSInitialization:
    """Test SOS initialization."""

    def test_default_parameters(self) -> None:
        """Test default parameter values."""
        detector = SOS()
        assert detector.perplexity == 30
        assert detector.metric == 'euclidean'
        assert detector.eps == 1e-5

    def test_custom_parameters(self) -> None:
        """Test custom parameter values."""
        detector = SOS(perplexity=15, metric='manhattan', eps=1e-6)
        assert detector.perplexity == 15
        assert detector.metric == 'manhattan'
        assert detector.eps == 1e-6


class TestSOSTransformations:
    """Test individual transformation methods."""

    def test_x2d_euclidean(self, simple_data: np.ndarray) -> None:
        """Test Euclidean dissimilarity matrix computation."""
        detector = SOS(metric='euclidean')
        D = detector.x2d(simple_data)
        assert D.shape == (5, 5)
        assert np.allclose(D.diagonal(), 0)  # Distance to self is 0
        assert np.allclose(D, D.T)  # Symmetric matrix

    def test_x2d_none_metric(self) -> None:
        """Test with pre-computed dissimilarity matrix."""
        D_input = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        detector = SOS(metric='none')
        D = detector.x2d(D_input)
        assert_array_equal(D, D_input)

    def test_x2d_none_metric_invalid_shape(self) -> None:
        """Non-square matrix should raise ValueError."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        detector = SOS(metric='none')
        with pytest.raises(ValueError, match='square dissimilarity matrix'):
            detector.x2d(X)

    def test_d2a_affinity_properties(self, simple_data: np.ndarray) -> None:
        """Test affinity matrix properties."""
        detector = SOS()
        D = detector.x2d(simple_data)
        A = detector.d2a(D)
        assert A.shape == D.shape
        assert np.all(A >= 0)  # All affinities non-negative

    def test_a2b_sum_to_one(self, simple_data: np.ndarray) -> None:
        """Test binding probabilities sum to 1."""
        detector = SOS()
        D = detector.x2d(simple_data)
        A = detector.d2a(D)
        B = detector.a2b(A)
        row_sums = B.sum(axis=1)
        assert_array_almost_equal(row_sums, np.ones(len(simple_data)))

    def test_b2o_range(self, simple_data: np.ndarray) -> None:
        """Test outlier probabilities in [0, 1]."""
        detector = SOS()
        D = detector.x2d(simple_data)
        A = detector.d2a(D)
        B = detector.a2b(A)
        O = detector.b2o(B)
        assert np.all(O >= 0)
        assert np.all(O <= 1)


class TestSOSPredict:
    """Test main predict method."""

    def test_predict_output_shape(self, simple_data: np.ndarray) -> None:
        """Test predict returns correct shape."""
        detector = SOS()
        O = detector.predict(simple_data)
        assert O.shape == (len(simple_data),)

    def test_predict_identifies_outlier(self, simple_data: np.ndarray) -> None:
        """Test that clear outlier gets high score."""
        detector = SOS(perplexity=2)
        O = detector.predict(simple_data)
        # Point [10, 10] should have highest outlier score
        assert np.argmax(O) == 4

    @pytest.mark.parametrize('perplexity', [5, 10, 20, 30])
    def test_predict_different_perplexities(
        self, simple_data: np.ndarray, perplexity: float
    ) -> None:
        """Test predict works with different perplexities."""
        detector = SOS(perplexity=perplexity)
        O = detector.predict(simple_data)
        assert O.shape == (len(simple_data),)
        assert np.all(O >= 0)
        assert np.all(O <= 1)

    def test_predict_single_point(self) -> None:
        """Test with single data point."""
        X = np.array([[1, 2]])
        detector = SOS()
        O = detector.predict(X)
        assert O.shape == (1,)

    def test_predict_identical_points(self) -> None:
        """Test with all identical points."""
        X = np.ones((5, 3))
        detector = SOS()
        O = detector.predict(X)
        # All points identical, similar scores
        assert np.allclose(O, O[0], rtol=0.1)


class TestSOSMetrics:
    """Test different distance metrics."""

    def test_euclidean_metric(self, simple_data: np.ndarray) -> None:
        """Test Euclidean metric."""
        detector = SOS(metric='euclidean')
        O = detector.predict(simple_data)
        assert O.shape == (len(simple_data),)

    @pytest.mark.parametrize('metric', ['cityblock', 'cosine', 'chebyshev'])
    def test_scipy_metrics(self, simple_data: np.ndarray, metric: str) -> None:
        """Test scipy distance metrics."""
        pytest.importorskip('scipy')
        detector = SOS(metric=metric)
        O = detector.predict(simple_data)
        assert O.shape == (len(simple_data),)


class TestSOSSklearnCompatibility:
    """Test scikit-learn compatibility."""

    def test_fit_returns_self(self, simple_data: np.ndarray) -> None:
        """Test fit returns self for chaining."""
        detector = SOS()
        result = detector.fit(simple_data)
        assert result is detector

    def test_fit_predict_chain(self, simple_data: np.ndarray) -> None:
        """Test fit().predict() chaining."""
        detector = SOS()
        O = detector.fit(simple_data).predict(simple_data)
        assert O.shape == (len(simple_data),)


class TestSOSEdgeCases:
    """Test edge cases and numerical stability."""

    def test_high_dimensional_data(self, rng: np.random.Generator) -> None:
        """Test with high-dimensional data."""
        X = rng.standard_normal((20, 100))
        detector = SOS()
        O = detector.predict(X)
        assert O.shape == (20,)

    def test_large_dataset(self, rng: np.random.Generator) -> None:
        """Test with larger dataset."""
        X = rng.standard_normal((500, 10))
        detector = SOS()
        O = detector.predict(X)
        assert O.shape == (500,)

    def test_very_large_distances(self) -> None:
        """Test numerical stability with large distances."""
        X = np.array([[0, 0], [1e6, 1e6]], dtype=float)
        detector = SOS()
        O = detector.predict(X)
        assert not np.any(np.isnan(O))
        assert not np.any(np.isinf(O))
