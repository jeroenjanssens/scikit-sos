from __future__ import annotations

import inspect
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from sklearn.base import BaseEstimator, OutlierMixin
    from sklearn.utils.validation import check_is_fitted

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

if HAS_SKLEARN:
    try:
        from sklearn.utils.validation import validate_data as _validate_data_fn

        _USE_STANDALONE_VALIDATE = True
    except ImportError:
        _USE_STANDALONE_VALIDATE = False

    try:
        from sklearn.utils._param_validation import Interval

        _HAS_PARAM_VALIDATION = True
    except ImportError:
        _HAS_PARAM_VALIDATION = False
else:
    _USE_STANDALONE_VALIDATE = False
    _HAS_PARAM_VALIDATION = False

# Type aliases
FloatArray = NDArray[np.floating[Any]]


class _BaseEstimator:
    """Minimal sklearn-compatible base when sklearn is not installed."""

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name in inspect.signature(self.__init__).parameters:  # type: ignore[misc]
            if name == 'self':
                continue
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params: Any) -> _BaseEstimator:
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f'Invalid parameter {key!r}')
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        sig = inspect.signature(self.__init__)  # type: ignore[misc]
        parts = []
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            value = getattr(self, name)
            if param.default is not inspect.Parameter.empty and value == param.default:
                continue
            parts.append(f'{name}={value!r}')
        return f'{cls_name}({", ".join(parts)})'


def _validate_data_fallback(
    estimator: Any,
    X: ArrayLike,
    *,
    reset: bool = True,
) -> FloatArray:
    X_arr: FloatArray = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        raise ValueError(
            'Expected 2D array, got 1D array instead.\n'
            'Reshape your data either using array.reshape(-1, 1) if your data has '
            'a single feature or array.reshape(1, -1) if it contains a single sample.'
        )
    if not np.all(np.isfinite(X_arr)):
        if np.any(np.isnan(X_arr)):
            raise ValueError('Input X contains NaN.')
        raise ValueError('Input X contains infinity.')
    if reset:
        estimator.n_features_in_ = X_arr.shape[1]
    else:
        if X_arr.shape[1] != estimator.n_features_in_:
            raise ValueError(
                f'X has {X_arr.shape[1]} features, but {type(estimator).__name__} is expecting '
                f'{estimator.n_features_in_} features as input.'
            )
    return X_arr


_Base = (BaseEstimator, OutlierMixin) if HAS_SKLEARN else (_BaseEstimator,)


class SOS(*_Base):  # type: ignore[misc]
    """Stochastic Outlier Selection.

    SOS is an unsupervised outlier detection algorithm that uses the concept
    of affinity to compute an outlier probability for each data point.
    It converts a dissimilarity matrix to an affinity matrix, then to binding
    probabilities, and finally to outlier probabilities.

    Parameters
    ----------
    perplexity : float, default=30.0
        Target perplexity for the affinity calculation. Controls the effective
        number of neighbors. Must be positive. Typical values are between 5
        and 50.

    metric : str, default='euclidean'
        Distance metric for computing the dissimilarity matrix. Supports
        'euclidean' (built-in), 'none' (pre-computed dissimilarity matrix),
        or any metric supported by ``scipy.spatial.distance.pdist``.

    eps : float, default=1e-5
        Convergence tolerance for the perplexity binary search. Must be
        positive.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during :meth:`fit`. Defined only when ``X``
        has feature names that are all strings.

    See Also
    --------
    sklearn.neighbors.LocalOutlierFactor : Unsupervised outlier detection
        using the Local Outlier Factor (LOF).
    sklearn.ensemble.IsolationForest : Isolation Forest anomaly detection.

    References
    ----------
    J.H.M. Janssens, F. Huszar, E.O. Postma, and H.J. van den Herik.
    Stochastic Outlier Selection. Technical Report TiCC TR 2012-001,
    Tilburg University, Tilburg, the Netherlands, 2012.

    Examples
    --------
    >>> import numpy as np
    >>> from sksos import SOS
    >>> X = np.array([[0, 0], [1, 1], [0, 1], [1, 0], [10, 10]])
    >>> detector = SOS(perplexity=2)
    >>> detector.fit(X)
    SOS(perplexity=2)
    >>> scores = detector.predict(X)
    >>> scores.shape
    (5,)
    """

    if _HAS_PARAM_VALIDATION:
        _parameter_constraints: dict[str, list[Any]] = {
            'perplexity': [Interval(Real, 0, None, closed='neither')],
            'metric': [str],
            'eps': [Interval(Real, 0, None, closed='neither')],
        }

    def __init__(
        self,
        perplexity: float = 30.0,
        metric: str = 'euclidean',
        eps: float = 1e-5,
    ) -> None:
        self.perplexity = perplexity
        self.metric = metric
        self.eps = eps

    def _validate_params_manual(self) -> None:
        """Validate parameters when sklearn parameter validation is unavailable."""
        if not isinstance(self.perplexity, (int, float)):
            raise TypeError(f'perplexity must be a number, got {type(self.perplexity).__name__}')
        if self.perplexity <= 0:
            raise ValueError(f'perplexity must be positive, got {self.perplexity}')
        if not isinstance(self.eps, (int, float)):
            raise TypeError(f'eps must be a number, got {type(self.eps).__name__}')
        if self.eps <= 0:
            raise ValueError(f'eps must be positive, got {self.eps}')
        if not isinstance(self.metric, str):
            raise TypeError(f'metric must be a string, got {type(self.metric).__name__}')

    def fit(self, X: ArrayLike, y: None = None) -> SOS:
        """Fit the model.

        Validates parameters and input data, and stores feature metadata.
        SOS is a transductive algorithm, so fitting only validates and
        records metadata.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : None
            Not used, present for API consistency.

        Returns
        -------
        self : SOS
            Fitted estimator.
        """
        if _HAS_PARAM_VALIDATION:
            self._validate_params()
        else:
            self._validate_params_manual()

        if HAS_SKLEARN:
            if _USE_STANDALONE_VALIDATE:
                X_validated: FloatArray = _validate_data_fn(
                    self,
                    X,
                    accept_sparse=False,
                    dtype=np.float64,
                    ensure_2d=True,
                    ensure_all_finite=True,
                    copy=False,
                )
            else:
                X_validated = self._validate_data(
                    X,
                    accept_sparse=False,
                    dtype=np.float64,
                    ensure_2d=True,
                    force_all_finite=True,
                    copy=False,
                )
        else:
            X_validated = _validate_data_fallback(self, X, reset=True)

        metric_lower = self.metric.lower()
        if metric_lower == 'none':
            n, d = X_validated.shape
            if n != d:
                raise ValueError(
                    "If you specify 'none' as the metric, the data set "
                    'should be a square dissimilarity matrix'
                )

        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict outlier probabilities for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Outlier probability for each sample. Values are in [0, 1],
            where higher values indicate more outlier-like samples.
        """
        if HAS_SKLEARN:
            check_is_fitted(self)
            if _USE_STANDALONE_VALIDATE:
                X_validated: FloatArray = _validate_data_fn(
                    self,
                    X,
                    accept_sparse=False,
                    dtype=np.float64,
                    ensure_2d=True,
                    ensure_all_finite=True,
                    reset=False,
                )
            else:
                X_validated = self._validate_data(
                    X,
                    accept_sparse=False,
                    dtype=np.float64,
                    ensure_2d=True,
                    force_all_finite=True,
                    reset=False,
                )
        else:
            if not hasattr(self, 'n_features_in_'):
                raise AttributeError(
                    f'This {type(self).__name__} instance is not fitted yet. '
                    f"Call 'fit' with appropriate arguments before using this estimator."
                )
            X_validated = _validate_data_fallback(self, X, reset=False)

        D = self._x2d(X_validated)
        A = self._d2a(D)
        B = self._a2b(A)
        O = self._b2o(B)
        return O

    def fit_predict(self, X: ArrayLike, y: None = None) -> FloatArray:
        """Fit the model and predict outlier probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : None
            Not used, present for API consistency.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Outlier probability for each sample.
        """
        return self.fit(X, y).predict(X)

    def score_samples(self, X: ArrayLike) -> FloatArray:
        """Compute outlier scores for each sample.

        Alias for :meth:`predict`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Outlier probability for each sample.
        """
        return self.predict(X)

    def decision_function(self, X: ArrayLike) -> FloatArray:
        """Compute the decision function for each sample.

        For SOS, this returns the outlier probability. Higher values
        indicate outliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Decision function values.
        """
        return self.predict(X)

    def _x2d(self, X: FloatArray) -> FloatArray:
        """Compute dissimilarity matrix."""
        (n, d) = X.shape
        metric = self.metric.lower()
        if metric == 'none':
            if n != d:
                raise ValueError(
                    "If you specify 'none' as the metric, the data set "
                    'should be a square dissimilarity matrix'
                )
            D = X
        elif metric == 'euclidean':
            sumX = np.sum(np.square(X), 1)
            D = np.sqrt(np.abs(np.add(np.add(-2 * np.dot(X, X.T), sumX).T, sumX)))
        else:
            try:
                from scipy.spatial import distance
            except ImportError as err:
                raise ImportError(
                    'Please install scipy if you wish to use a metric '
                    "other than 'euclidean' or 'none'"
                ) from err
            else:
                D = distance.squareform(distance.pdist(X, metric))
        return D

    def _d2a(self, D: FloatArray) -> FloatArray:
        """Return affinity matrix.

        Performs a binary search to get affinities in such a way that each
        conditional Gaussian has the same perplexity.
        """
        (n, _) = D.shape
        A = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(self.perplexity)

        for i in range(n):
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))]
            (H, thisA) = _get_perplexity(Di, beta[i])

            Hdiff = H - logU
            tries = 0
            while (np.isnan(Hdiff) or np.abs(Hdiff) > self.eps) and tries < 5000:
                if np.isnan(Hdiff):
                    beta[i] = beta[i] / 10.0
                elif Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.0
                    else:
                        beta[i] = (beta[i] + betamax) / 2.0
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.0
                    else:
                        beta[i] = (beta[i] + betamin) / 2.0
                (H, thisA) = _get_perplexity(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            A[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))] = thisA

        return A

    def _a2b(self, A: FloatArray) -> FloatArray:
        """Convert affinity to binding probability."""
        B: FloatArray = A / A.sum(axis=1)[:, np.newaxis]
        return B

    def _b2o(self, B: FloatArray) -> FloatArray:
        """Convert binding probability to outlier probability."""
        O: FloatArray = np.prod(1 - B, 0)
        return O


def _get_perplexity(
    D: FloatArray,
    beta: float | FloatArray,
) -> tuple[float | FloatArray, FloatArray]:
    """Compute the perplexity and the A-row for a specific value of the
    precision of a Gaussian distribution.
    """
    A = np.exp(-D * beta)
    sumA = sum(A)
    H = np.log(sumA) + beta * np.sum(D * A) / sumA
    return H, A
