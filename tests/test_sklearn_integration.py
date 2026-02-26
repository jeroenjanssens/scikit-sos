"""Tests for scikit-learn compatibility."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sksos import SOS


class TestSklearnCompatibility:
    """Test scikit-learn API compliance."""

    def test_get_params(self) -> None:
        detector = SOS(perplexity=15, metric='manhattan', eps=1e-6)
        params = detector.get_params()
        assert params == {'perplexity': 15, 'metric': 'manhattan', 'eps': 1e-6}

    def test_set_params(self) -> None:
        detector = SOS()
        detector.set_params(perplexity=20)
        assert detector.perplexity == 20

    def test_clone(self, simple_data: np.ndarray) -> None:
        detector = SOS(perplexity=15)
        detector.fit(simple_data)
        cloned = clone(detector)
        assert cloned.perplexity == 15
        assert not hasattr(cloned, 'n_features_in_')

    def test_repr_default(self) -> None:
        assert repr(SOS()) == 'SOS()'

    def test_repr_custom(self) -> None:
        assert repr(SOS(perplexity=20)) == 'SOS(perplexity=20)'


class TestFitValidation:
    """Test fit() input validation."""

    def test_fit_stores_n_features(self, simple_data: np.ndarray) -> None:
        detector = SOS()
        detector.fit(simple_data)
        assert detector.n_features_in_ == 2

    def test_fit_with_pandas(self) -> None:
        pd = pytest.importorskip('pandas')
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        detector = SOS()
        detector.fit(df)
        assert detector.n_features_in_ == 2
        assert_array_equal(detector.feature_names_in_, ['a', 'b'])

    def test_fit_rejects_nan(self) -> None:
        X = np.array([[1, 2], [np.nan, 4]], dtype=float)
        detector = SOS()
        with pytest.raises(ValueError, match='NaN'):
            detector.fit(X)

    def test_fit_rejects_inf(self) -> None:
        X = np.array([[1, 2], [np.inf, 4]], dtype=float)
        detector = SOS()
        with pytest.raises(ValueError, match='infinity'):
            detector.fit(X)

    def test_fit_rejects_1d(self) -> None:
        X = np.array([1, 2, 3], dtype=float)
        detector = SOS()
        with pytest.raises(ValueError, match='2D'):
            detector.fit(X)

    def test_fit_accepts_list(self) -> None:
        X = [[1, 2], [3, 4], [5, 6]]
        detector = SOS()
        detector.fit(X)
        assert detector.n_features_in_ == 2


class TestPredictValidation:
    """Test predict() input validation."""

    def test_predict_before_fit_raises(self) -> None:
        detector = SOS()
        with pytest.raises(NotFittedError):
            detector.predict(np.array([[1, 2]]))

    def test_predict_wrong_n_features(self, simple_data: np.ndarray) -> None:
        detector = SOS()
        detector.fit(simple_data)
        with pytest.raises(ValueError, match='features'):
            detector.predict(np.array([[1, 2, 3]]))

    def test_predict_with_nan_raises(self, simple_data: np.ndarray) -> None:
        detector = SOS()
        detector.fit(simple_data)
        X_bad = np.array([[1, np.nan]])
        with pytest.raises(ValueError, match='NaN'):
            detector.predict(X_bad)


class TestParameterValidation:
    """Test parameter validation."""

    def test_negative_perplexity_raises(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=float)
        with pytest.raises((ValueError, TypeError)):
            SOS(perplexity=-1).fit(X)

    def test_zero_perplexity_raises(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=float)
        with pytest.raises((ValueError, TypeError)):
            SOS(perplexity=0).fit(X)

    def test_negative_eps_raises(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=float)
        with pytest.raises((ValueError, TypeError)):
            SOS(eps=-1).fit(X)

    def test_invalid_metric_type_raises(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=float)
        with pytest.raises((ValueError, TypeError)):
            SOS(metric=123).fit(X)  # type: ignore[arg-type]


class TestConvenienceMethods:
    """Test convenience methods."""

    def test_fit_predict(self, simple_data: np.ndarray) -> None:
        scores1 = SOS(perplexity=2).fit_predict(simple_data)
        scores2 = SOS(perplexity=2).fit(simple_data).predict(simple_data)
        assert_array_equal(scores1, scores2)

    def test_score_samples_equals_predict(self, simple_data: np.ndarray) -> None:
        detector = SOS(perplexity=2)
        detector.fit(simple_data)
        assert_array_equal(detector.score_samples(simple_data), detector.predict(simple_data))

    def test_decision_function_equals_predict(self, simple_data: np.ndarray) -> None:
        detector = SOS(perplexity=2)
        detector.fit(simple_data)
        assert_array_equal(detector.decision_function(simple_data), detector.predict(simple_data))


class TestPipelineIntegration:
    """Test integration with sklearn Pipeline."""

    def test_in_pipeline(self, simple_data: np.ndarray) -> None:
        pipe = Pipeline([('scaler', StandardScaler()), ('sos', SOS(perplexity=2))])
        pipe.fit(simple_data)
        scores = pipe.predict(simple_data)
        assert scores.shape == (len(simple_data),)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_pipeline_set_params(self) -> None:
        pipe = Pipeline([('sos', SOS())])
        pipe.set_params(sos__perplexity=20)
        assert pipe.named_steps['sos'].perplexity == 20

    def test_pipeline_get_params(self) -> None:
        pipe = Pipeline([('sos', SOS(perplexity=25))])
        params = pipe.get_params()
        assert params['sos__perplexity'] == 25
