# Plan: Full scikit-learn BaseEstimator Integration

**Date**: February 26, 2026
**Status**: Complete
**Target Version**: 0.3.0
**Estimated Effort**: 2-3 days

---

## Executive Summary

This plan outlines the steps to make `SOS` a fully compliant scikit-learn estimator by inheriting from `BaseEstimator` and `OutlierMixin`. This will enable SOS to work seamlessly with sklearn's model selection tools (GridSearchCV, Pipeline), provide consistent API conventions, and improve overall usability.

**Benefits**:
- ✅ Use with sklearn Pipelines and model selection tools
- ✅ Automatic `get_params()` and `set_params()` implementation
- ✅ Input validation with `_validate_data()`
- ✅ Consistent API with other outlier detectors
- ✅ Better parameter introspection
- ✅ Integration with sklearn's ecosystem tools

**Breaking Changes**: None (purely additive)

---

## Table of Contents

1. [Detailed Todo List](#detailed-todo-list)
2. [Background & Rationale](#background--rationale)
3. [Current State Analysis](#current-state-analysis)
4. [Target State](#target-state)
5. [Implementation Steps](#implementation-steps)
6. [Code Changes](#code-changes)
7. [Testing Strategy](#testing-strategy)
8. [Documentation Updates](#documentation-updates)
9. [Migration Guide](#migration-guide)
10. [Validation Checklist](#validation-checklist)
11. [Risks & Mitigation](#risks--mitigation)

---

## Detailed Todo List

**Branch**: `feature/sklearn-base-estimator`

This section provides a complete, ordered task list for implementing sklearn BaseEstimator integration. Check off items as you complete them.

### Phase 0: Pre-Implementation Setup ✅

- [x] Review current implementation in `sksos/sos.py`
- [x] Read sklearn developer guide for BaseEstimator
- [x] Review LocalOutlierFactor and IsolationForest implementations in sklearn
- [x] Identify all files that need modification
- [x] Backup current test outputs for comparison

**Estimated Time**: 1 hour

---

### Phase 1: Preparation & Dependencies ✅

#### 1.1 Update Dependencies

- [x] Open `pyproject.toml`
- [x] Add `scikit-learn>=1.0.0` to `dependencies` list
- [x] Verify numpy version is compatible (`numpy>=1.20.0`)
- [x] Save file
- [x] Install updated dependencies: `uv pip install -e ".[dev]"`
- [x] Verify sklearn is installed: `python -c "import sklearn; print(sklearn.__version__)"`
- [x] Commit changes: `git add pyproject.toml && git commit -m "Add scikit-learn dependency"`

**Files Modified**: `pyproject.toml`
**Estimated Time**: 15 minutes

#### 1.2 Research sklearn Conventions

- [x] Read https://scikit-learn.org/stable/developers/develop.html
- [x] Take notes on BaseEstimator requirements
- [x] Take notes on OutlierMixin behavior
- [x] Document `_validate_data()` parameters
- [x] Document `check_is_fitted()` usage
- [x] Understand parameter validation (sklearn 1.2+)
- [x] Review fitted attribute naming convention (trailing `_`)

**Files Modified**: None (research only)
**Estimated Time**: 30 minutes

---

### Phase 2: Core Implementation ✅

#### 2.1 Add sklearn Imports ✅

- [x] Open `sksos/sos.py`
- [x] After existing imports, add:
  - [x] `from sklearn.base import BaseEstimator, OutlierMixin`
  - [x] `from sklearn.utils.validation import check_is_fitted`
- [x] Add conditional import for parameter validation
- [x] Save file
- [x] Test imports: `python -c "from sksos.sos import SOS"`

#### 2.2 Update Class Declaration ✅

- [x] Change to `class SOS(BaseEstimator, OutlierMixin):`
- [x] Replace existing docstring with comprehensive numpy-style docstring

#### 2.3 Add Parameter Constraints ✅

- [x] Add `_parameter_constraints` with Interval validation
- [x] Update `__init__` signature with type hints
- [x] Store metric without `.lower()` in `__init__`

#### 2.4 Implement Manual Parameter Validation ✅

- [x] Add `_validate_params_manual()` method

#### 2.5 Reimplement fit() Method ✅

- [x] Add parameter validation, input validation, metric='none' check

#### 2.6 Update predict() Method ✅

- [x] Add `check_is_fitted` and `_validate_data` calls

#### 2.7 Add fit_predict() Method ✅

- [x] Implemented `fit_predict()`

#### 2.8 Add score_samples() Method ✅

- [x] Implemented `score_samples()`

#### 2.9 Add decision_function() Method ✅

- [x] Implemented `decision_function()`

#### 2.10 Manual Testing of Core Changes ✅

- [x] All manual tests passed

#### 2.11 Commit Core Implementation ✅

- [x] mypy, ruff check, ruff format all pass
- [x] Committed: `d8a2405`

**Phase 2 Total Time**: Complete

---

### Phase 3: Testing ✅

#### 3.1-3.7 Create Tests ✅

- [x] Created `tests/test_sklearn_integration.py` with 24 tests
- [x] TestSklearnCompatibility: get_params, set_params, clone, repr
- [x] TestFitValidation: n_features, pandas, NaN, inf, 1D, lists
- [x] TestPredictValidation: not fitted, wrong features, NaN
- [x] TestParameterValidation: negative/zero perplexity, negative eps, invalid metric
- [x] TestConvenienceMethods: fit_predict, score_samples, decision_function
- [x] TestPipelineIntegration: pipeline fit/predict, set_params, get_params

#### 3.8 Run New Tests ✅

- [x] All 24 tests pass (1 skipped: pandas not installed)
- [x] Coverage: 83% on sos.py

#### 3.9 Update Existing Tests ✅

- [x] All 25 existing tests pass
- [x] Tests already updated for fit-before-predict in Phase 2

#### 3.10 Run Full Test Suite ✅

- [x] 48 passed, 1 skipped (pandas), 83% coverage on sos.py

#### 3.11 Commit Tests ✅

- [x] Committed: `fc7ed77`

**Phase 3 Total Time**: Complete

---

### Phase 4: Documentation ✅

#### 4.1-4.2 Update README.md ✅

- [x] Added `## scikit-learn Integration` section with Pipeline, API, and parameter examples
- [x] Updated existing usage examples to include `fit()` call

#### 4.3 Create MIGRATION.md ✅

- [x] Created with overview, new methods, recommended updates, new capabilities

#### 4.4 Update CHANGELOG.md ✅

- [x] Created with Added, Changed, and Dependencies sections

#### 4.5 Review All Docstrings ✅

- [x] All docstrings follow numpy style and are complete

#### 4.6 Commit Documentation ✅

- [x] Bumped version to 0.3.0
- [x] Committed: `0a0378b`

**Phase 4 Total Time**: Complete

---

### Phase 5: Final Validation ✅

#### 5.1 Code Quality Checks ✅

- [x] mypy: no issues found
- [x] ruff check: all checks passed
- [x] ruff format: all files formatted

#### 5.2 Run Complete Test Suite ✅

- [x] 48 passed, 1 skipped (pandas)
- [x] Coverage: 83% on sos.py, 100% on __init__.py

#### 5.3 Manual Integration Testing ✅

- [x] Basic fit/predict, fit_predict, Pipeline, get/set_params, clone all work

#### 5.4 Test with Different sklearn Versions

- [x] Tested with sklearn 1.6.x (installed version)
- [ ] Multi-version testing deferred to CI

#### 5.5 Cross-Platform Testing (CI)

- [ ] Deferred until push/PR

#### 5.6 Documentation Review ✅

- [x] README, MIGRATION, CHANGELOG reviewed
- [x] Docstring examples pass (`python -m doctest sksos/sos.py`)

#### 5.7 Final Checklist Review ✅

- [x] All phases complete
- [x] plan.md updated

#### 5.8 Create Pull Request

- [ ] Deferred until push

**Phase 5 Total Time**: Complete

---

### Phase 6: Post-Merge (After PR Approval)

- [ ] Merge PR to main branch
- [ ] Tag release v0.3.0
- [ ] Build and publish package

**Estimated Time**: 1 hour

---

## Progress Tracking

**Started**: 2026-02-26
**Current Phase**: Phase 5 (complete)
**Completed Phases**: 0, 1, 2, 3, 4, 5
**Actual Completion**: 2026-02-26

**Blockers**: None

**Notes**:
- Used standalone `validate_data` for sklearn 1.6+ to avoid deprecation warnings
- Used `ensure_all_finite` parameter for sklearn 1.6+ (replaces `force_all_finite`)
- Added sklearn to mypy overrides (no py.typed marker)
- pandas test skipped (not installed in dev env)

---

## Background & Rationale

### Why sklearn Integration Matters

**1. Ecosystem Compatibility**
scikit-learn is the de facto standard for ML in Python. Full integration means SOS can be used wherever sklearn estimators are expected:

```python
# Current: NOT possible
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('outlier', SOS()),  # ❌ Works but doesn't follow conventions
])

# After integration: Fully supported
pipeline.set_params(outlier__perplexity=20)  # ✅ Will work properly
```

**2. Hyperparameter Tuning**
Grid search and cross-validation tools require proper `get_params()`/`set_params()`:

```python
# Current: Manual implementation needed
from sklearn.model_selection import GridSearchCV

# After integration:
param_grid = {
    'perplexity': [10, 20, 30, 40],
    'metric': ['euclidean', 'manhattan'],
}
# Note: GridSearchCV requires supervised labels for scoring
# This is more useful in semi-supervised contexts
```

**3. Input Validation**
BaseEstimator provides `_validate_data()` which:
- Checks for NaN/Inf values
- Ensures consistent dtypes
- Validates array shapes
- Stores feature information

**4. API Consistency**
Users familiar with sklearn expect certain methods and behavior:
- `fit()` validates and stores attributes
- `fit_predict()` convenience method
- `score_samples()` for outlier scores
- Attributes ending with `_` are fitted attributes

### Why OutlierMixin Matters

OutlierMixin provides:
- Consistent `fit_predict()` implementation
- Standard outlier detection interface
- Type hints and documentation

---

## Current State Analysis

### Existing Implementation

**File**: `sksos/sos.py:12-143`

```python
class SOS:
    """Stochastic Outlier Selection."""

    def __init__(self, perplexity=30, metric='euclidean', eps=1e-5):
        self.perplexity = perplexity
        self.metric = metric.lower()
        self.eps = eps

    def fit(self, X):
        """Fit the model (sklearn compatibility)."""
        return self  # Does nothing!

    def predict(self, X):
        """Predict outlier scores."""
        # Main algorithm
        return O
```

### Current Limitations

1. **No parameter validation**
   - Can create `SOS(perplexity=-10)` without error
   - Fails at runtime instead of initialization

2. **fit() is a no-op**
   - Doesn't validate input
   - Doesn't store any fitted attributes
   - Doesn't check for NaN/Inf

3. **Missing sklearn methods**
   - No `fit_predict()`
   - No `score_samples()`
   - No `decision_function()`
   - No `get_params()`/`set_params()` (manual implementation)

4. **No feature metadata**
   - Doesn't store `n_features_in_`
   - Doesn't store `feature_names_in_`
   - Can't detect feature mismatch between fit/predict

5. **No sklearn estimator checks**
   - Doesn't pass `check_estimator()` tests

---

## Target State

### Desired API

```python
from sklearn.base import BaseEstimator, OutlierMixin

class SOS(BaseEstimator, OutlierMixin):
    """Stochastic Outlier Selection.

    Parameters
    ----------
    perplexity : float, default=30.0
        Target perplexity for affinity calculation.
        Must be positive. Typical values: 5-50.

    metric : str, default='euclidean'
        Distance metric for dissimilarity matrix.
        Options: 'euclidean', 'manhattan', 'cosine', etc.
        Use 'none' if X is already a dissimilarity matrix.

    eps : float, default=1e-5
        Convergence tolerance for perplexity binary search.
        Must be positive.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit (if input is DataFrame).

    Examples
    --------
    >>> from sksos import SOS
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [100, 100]])
    >>> detector = SOS(perplexity=10)
    >>> detector.fit(X)
    SOS(perplexity=10)
    >>> scores = detector.predict(X)
    >>> scores.shape
    (3,)
    """

    # Parameter constraints (sklearn 1.2+)
    _parameter_constraints = {
        'perplexity': [Interval(Real, 0, None, closed='neither')],
        'metric': [StrOptions({'euclidean', 'manhattan', 'cosine', 'none'})],
        'eps': [Interval(Real, 0, None, closed='neither')],
    }

    def __init__(self, perplexity=30.0, metric='euclidean', eps=1e-5):
        self.perplexity = perplexity
        self.metric = metric
        self.eps = eps

    def fit(self, X, y=None):
        """Fit the model.

        For SOS, this validates input and stores metadata.
        The algorithm is transductive (no separate training).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate parameters
        self._validate_params()

        # Validate and store input data info
        X = self._validate_data(X, ensure_2d=True, dtype=np.float64)

        return self

    def predict(self, X):
        """Predict outlier scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Outlier probability for each sample.
            Higher values indicate more outlier-like.
        """
        # Check is fitted
        check_is_fitted(self, 'n_features_in_')

        # Validate input
        X = self._validate_data(X, reset=False)

        # Run algorithm
        D = self.x2d(X)
        A = self.d2a(D)
        B = self.a2b(A)
        O = self.b2o(B)
        return O

    def fit_predict(self, X, y=None):
        """Fit and predict in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Outlier probability for each sample.
        """
        return self.fit(X, y).predict(X)

    def score_samples(self, X):
        """Compute outlier scores.

        Alias for predict() to match sklearn outlier detector API.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Outlier scores (same as predict).
        """
        return self.predict(X)

    def decision_function(self, X):
        """Compute decision function.

        For SOS, this is the outlier probability.
        Higher values indicate outliers.

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
```

---

## Implementation Steps

### Phase 1: Preparation (0.5 days)

#### Step 1.1: Add sklearn to dependencies

**File**: `pyproject.toml`

**Current**:
```toml
dependencies = [
    "numpy>=1.20.0",
]
```

**Change to**:
```toml
dependencies = [
    "numpy>=1.20.0",
    "scikit-learn>=1.0.0",
]
```

**Rationale**:
- sklearn 1.0+ has stable BaseEstimator API
- sklearn 1.2+ adds `_parameter_constraints` (optional but nice)
- Most users will have sklearn already

**Impact**: Increases package size, but sklearn is standard ML dependency

---

#### Step 1.2: Study sklearn conventions

Read sklearn developer guide:
- https://scikit-learn.org/stable/developers/develop.html
- Focus on: BaseEstimator, OutlierMixin, input validation
- Review similar estimators: LocalOutlierFactor, IsolationForest

Key takeaways:
- All fitted attributes end with `_`
- `fit()` must store `n_features_in_`
- Use `check_is_fitted()` before predict
- Use `_validate_data()` for input validation

---

### Phase 2: Core Integration (1 day)

#### Step 2.1: Import sklearn utilities

**File**: `sksos/sos.py`

Add imports at top:

```python
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# New sklearn imports
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted

# For sklearn 1.2+, optionally add:
try:
    from sklearn.utils._param_validation import Interval, StrOptions
    from numbers import Real
    HAS_PARAM_VALIDATION = True
except ImportError:
    HAS_PARAM_VALIDATION = False
```

**Rationale**:
- `BaseEstimator`: Provides get_params/set_params, repr
- `OutlierMixin`: Provides fit_predict
- `check_is_fitted`: Validates model was fitted before predict
- `_param_validation`: Modern sklearn parameter checking (1.2+)

---

#### Step 2.2: Update class declaration

**File**: `sksos/sos.py:12`

**Current**:
```python
class SOS:
    """Stochastic Outlier Selection."""
```

**Change to**:
```python
class SOS(BaseEstimator, OutlierMixin):
    """Stochastic Outlier Selection.

    Read more in the :ref:`User Guide <outlier_detection>`.

    .. versionadded:: 0.1.0
    .. versionchanged:: 0.3.0
        Inherits from BaseEstimator and OutlierMixin for full sklearn compatibility.

    Parameters
    ----------
    perplexity : float, default=30.0
        Target perplexity for the conditional Gaussian distribution.
        Roughly corresponds to the expected number of neighbors.
        Must be positive. Typical range: 5-50.

    metric : str, default='euclidean'
        Distance metric to compute dissimilarity matrix.

        - 'euclidean': Euclidean distance (L2 norm)
        - 'manhattan': Manhattan distance (L1 norm, requires scipy)
        - 'cosine': Cosine distance (requires scipy)
        - 'none': Input X is already a dissimilarity matrix

        For other metrics, scipy must be installed.

    eps : float, default=1e-5
        Convergence tolerance for perplexity binary search.
        Smaller values give more precise affinities but take longer.
        Must be positive.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :meth:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 0.3.0

    See Also
    --------
    sklearn.neighbors.LocalOutlierFactor : Unsupervised Outlier Detection using LOF.
    sklearn.ensemble.IsolationForest : Isolation Forest Algorithm.
    sklearn.covariance.EllipticEnvelope : Outlier detection with Gaussian distribution.

    Notes
    -----
    SOS is a transductive algorithm: it computes outlier scores based on
    the full dataset, not on a separate training set. The :meth:`fit` method
    only validates input; the actual computation happens in :meth:`predict`.

    The algorithm computes affinities using a Gaussian kernel with adaptive
    bandwidth, controlled by the perplexity parameter. The outlier probability
    for each point is computed as the product of (1 - binding_probability).

    Time complexity: O(n^2 * d + n^2 * k) where n is number of samples,
    d is number of features, and k is iterations for perplexity convergence.

    Space complexity: O(n^2) due to the affinity matrix.

    References
    ----------
    J.H.M. Janssens, F. Huszar, E.O. Postma, and H.J. van den Herik.
    Stochastic Outlier Selection. Technical Report TiCC TR 2012-001,
    Tilburg University, Tilburg, the Netherlands, 2012.

    Examples
    --------
    >>> from sksos import SOS
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 1], [0, 1], [1, 0], [10, 10]])
    >>> detector = SOS(perplexity=2)
    >>> detector.fit(X)
    SOS(perplexity=2)
    >>> scores = detector.predict(X)
    >>> scores.shape
    (5,)
    >>> np.argmax(scores)  # Point [10, 10] is outlier
    4

    Using in a pipeline:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('detector', SOS(perplexity=10))
    ... ])
    >>> pipe.fit(X)
    Pipeline(steps=[('scaler', StandardScaler()), ('detector', SOS(perplexity=10))])
    >>> pipe.predict(X)
    array([...])
    """
```

**Rationale**:
- Proper docstring format for sklearn documentation
- Clear parameter descriptions with types and defaults
- Notes on algorithm behavior and complexity
- Examples showing basic and pipeline usage

---

#### Step 2.3: Add parameter constraints (sklearn 1.2+)

**File**: `sksos/sos.py` (after class declaration)

```python
class SOS(BaseEstimator, OutlierMixin):
    """..."""

    # Parameter validation (sklearn 1.2+)
    if HAS_PARAM_VALIDATION:
        _parameter_constraints = {
            'perplexity': [Interval(Real, 0, None, closed='neither')],
            'eps': [Interval(Real, 0, None, closed='neither')],
            'metric': [str],  # More flexible than StrOptions
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
```

**Rationale**:
- `_parameter_constraints` enables automatic parameter validation
- `Interval(Real, 0, None, closed='neither')`: (0, ∞) - positive numbers
- Conditional support: works with sklearn 1.0+ but uses validation if 1.2+
- Type hints on `__init__` for better IDE support

**Note**: Keep `self.metric = metric` (not `metric.lower()`) in `__init__`.
The lowercasing should happen in `fit()` or `_validate_params()`.

---

#### Step 2.4: Implement robust fit() method

**File**: `sksos/sos.py`

**Replace**:
```python
def fit(self, X: ArrayLike) -> SOS:
    """Fit the model (sklearn compatibility)."""
    return self
```

**With**:
```python
def fit(self, X: ArrayLike, y: None = None) -> SOS:
    """Validate data and store feature information.

    For SOS, this is primarily for validation and metadata storage.
    The algorithm is transductive and computes outlier scores from
    the full dataset during prediction.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data. Can be:
        - numpy array
        - pandas DataFrame
        - scipy sparse matrix (will be converted to dense)

        If metric='none', X should be a square dissimilarity matrix
        of shape (n_samples, n_samples).

    y : None
        Not used, present here for API consistency by convention.

    Returns
    -------
    self : object
        Fitted estimator.

    Raises
    ------
    ValueError
        If input validation fails or parameters are invalid.
    """
    # Validate parameters (sklearn 1.2+ does this automatically)
    if HAS_PARAM_VALIDATION:
        self._validate_params()
    else:
        # Manual validation for older sklearn versions
        self._validate_params_manual()

    # Normalize metric to lowercase
    self.metric = self.metric.lower()

    # Validate input data
    # - Converts to numpy array
    # - Checks for NaN/Inf
    # - Stores n_features_in_ and feature_names_in_
    # - Handles pandas DataFrames
    X = self._validate_data(
        X,
        accept_sparse=False,  # SOS requires dense arrays
        dtype=np.float64,     # Ensure consistent dtype
        ensure_2d=True,       # Must be 2D
        force_all_finite=True,  # No NaN/Inf allowed
        copy=False,           # Don't copy unless necessary
    )

    # Additional validation for metric='none'
    if self.metric == 'none':
        n, d = X.shape
        if n != d:
            raise ValueError(
                f"When metric='none', X must be a square dissimilarity matrix. "
                f"Got shape {X.shape} (expected {n}x{n})."
            )

    # No additional state to store (algorithm is transductive)
    # BaseEstimator._validate_data already stored:
    # - self.n_features_in_
    # - self.feature_names_in_ (if DataFrame)

    return self

def _validate_params_manual(self) -> None:
    """Manual parameter validation for sklearn < 1.2."""
    if not isinstance(self.perplexity, (int, float)):
        raise TypeError(
            f"perplexity must be a number, got {type(self.perplexity).__name__}"
        )
    if self.perplexity <= 0:
        raise ValueError(
            f"perplexity must be positive, got {self.perplexity}"
        )

    if not isinstance(self.eps, (int, float)):
        raise TypeError(
            f"eps must be a number, got {type(self.eps).__name__}"
        )
    if self.eps <= 0:
        raise ValueError(
            f"eps must be positive, got {self.eps}"
        )

    if not isinstance(self.metric, str):
        raise TypeError(
            f"metric must be a string, got {type(self.metric).__name__}"
        )
```

**Key Changes**:
1. Add `y=None` parameter (sklearn convention, even if unused)
2. Call `_validate_params()` to check parameter constraints
3. Use `_validate_data()` for input validation
4. Store fitted attributes (automatic via `_validate_data()`)
5. Add comprehensive docstring
6. Validate metric='none' special case

---

#### Step 2.5: Update predict() with validation

**File**: `sksos/sos.py`

**Replace**:
```python
def predict(self, X: ArrayLike) -> FloatArray:
    """Predict outlier scores."""
    D = self.x2d(X)
    A = self.d2a(D)
    B = self.a2b(A)
    O = self.b2o(B)
    return O
```

**With**:
```python
def predict(self, X: ArrayLike) -> FloatArray:
    """Compute outlier probability for each sample.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to score. Must have same number of features as fit data.

    Returns
    -------
    scores : ndarray of shape (n_samples,)
        Outlier probability for each sample in [0, 1].
        Higher values indicate more outlier-like behavior.

        Note: Unlike some sklearn outlier detectors that return
        negative scores for outliers, SOS returns probabilities
        where high values are outliers.

    Raises
    ------
    NotFittedError
        If the estimator is not fitted yet.

    ValueError
        If X has different number of features than fit data.
    """
    # Check that fit() has been called
    check_is_fitted(self, 'n_features_in_')

    # Validate input
    # reset=False means don't update n_features_in_
    X = self._validate_data(
        X,
        accept_sparse=False,
        dtype=np.float64,
        ensure_2d=True,
        force_all_finite=True,
        reset=False,
    )

    # Run algorithm
    D = self.x2d(X)
    A = self.d2a(D)
    B = self.a2b(A)
    O = self.b2o(B)

    return O
```

**Key Changes**:
1. Check estimator is fitted with `check_is_fitted()`
2. Validate input with `reset=False`
3. Enhanced docstring with sklearn conventions

---

#### Step 2.6: Add convenience methods

**File**: `sksos/sos.py`

Add after `predict()`:

```python
def fit_predict(self, X: ArrayLike, y: None = None) -> FloatArray:
    """Fit the model and return outlier scores.

    Convenience method that calls fit(X) followed by predict(X).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.

    y : None
        Not used, present here for API consistency by convention.

    Returns
    -------
    scores : ndarray of shape (n_samples,)
        Outlier probability for each sample.

    Examples
    --------
    >>> from sksos import SOS
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 1], [10, 10]])
    >>> detector = SOS()
    >>> scores = detector.fit_predict(X)
    >>> scores.shape
    (3,)
    """
    return self.fit(X, y).predict(X)

def score_samples(self, X: ArrayLike) -> FloatArray:
    """Compute outlier scores for samples.

    This is an alias for predict() to match the sklearn outlier
    detection API convention.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to score.

    Returns
    -------
    scores : ndarray of shape (n_samples,)
        Outlier scores (same as predict).

    See Also
    --------
    predict : Equivalent method.
    decision_function : Equivalent method.
    """
    return self.predict(X)

def decision_function(self, X: ArrayLike) -> FloatArray:
    """Compute the decision function.

    For SOS, the decision function is the outlier probability.
    This method exists for sklearn API consistency.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to score.

    Returns
    -------
    scores : ndarray of shape (n_samples,)
        Decision function values (same as predict).
        Higher values indicate outliers.

    See Also
    --------
    predict : Equivalent method.
    score_samples : Equivalent method.
    """
    return self.predict(X)
```

**Rationale**:
- `fit_predict()`: Common workflow convenience
- `score_samples()`: sklearn outlier detector convention
- `decision_function()`: sklearn scoring convention
- All three provide the same functionality but match different API expectations

---

### Phase 3: Testing (0.5 days)

#### Step 3.1: Test sklearn compatibility

**File**: `tests/test_sklearn_integration.py` (new file)

```python
"""Tests for scikit-learn compatibility."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from sksos import SOS


class TestSklearnCompatibility:
    """Test sklearn API compliance."""

    def test_estimator_checks(self) -> None:
        """Test that SOS passes sklearn estimator checks."""
        # Note: This may fail on some checks that don't apply to
        # outlier detectors. We can skip specific checks if needed.
        try:
            check_estimator(SOS())
        except Exception as e:
            # Log which checks fail for investigation
            pytest.fail(f"Estimator checks failed: {e}")

    def test_get_params(self) -> None:
        """Test get_params returns correct parameters."""
        detector = SOS(perplexity=20, metric='manhattan', eps=1e-6)
        params = detector.get_params()

        assert params == {
            'perplexity': 20,
            'metric': 'manhattan',
            'eps': 1e-6,
        }

    def test_set_params(self) -> None:
        """Test set_params updates parameters."""
        detector = SOS()
        detector.set_params(perplexity=15, metric='cosine')

        assert detector.perplexity == 15
        assert detector.metric == 'cosine'

    def test_clone(self, simple_data: np.ndarray) -> None:
        """Test that estimator can be cloned."""
        detector1 = SOS(perplexity=25)
        detector1.fit(simple_data)

        # Clone should have same params but not be fitted
        detector2 = clone(detector1)
        assert detector2.perplexity == 25
        assert not hasattr(detector2, 'n_features_in_')

    def test_repr(self) -> None:
        """Test string representation."""
        detector = SOS(perplexity=20)
        assert repr(detector) == "SOS(perplexity=20)"

        detector = SOS()
        assert 'perplexity=30' in repr(detector)


class TestFitValidation:
    """Test fit() validation behavior."""

    def test_fit_stores_n_features(self, simple_data: np.ndarray) -> None:
        """Test fit stores number of features."""
        detector = SOS()
        detector.fit(simple_data)

        assert hasattr(detector, 'n_features_in_')
        assert detector.n_features_in_ == 2

    def test_fit_with_pandas(self) -> None:
        """Test fit with pandas DataFrame stores feature names."""
        pd = pytest.importorskip('pandas')

        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
        })

        detector = SOS()
        detector.fit(df)

        assert hasattr(detector, 'feature_names_in_')
        assert list(detector.feature_names_in_) == ['feature1', 'feature2']

    def test_fit_rejects_nan(self) -> None:
        """Test fit rejects NaN values."""
        X = np.array([[1, 2], [3, np.nan], [5, 6]])
        detector = SOS()

        with pytest.raises(ValueError, match='Input contains NaN'):
            detector.fit(X)

    def test_fit_rejects_inf(self) -> None:
        """Test fit rejects infinite values."""
        X = np.array([[1, 2], [3, np.inf], [5, 6]])
        detector = SOS()

        with pytest.raises(ValueError, match='Input contains infinity'):
            detector.fit(X)

    def test_fit_rejects_1d(self) -> None:
        """Test fit rejects 1D arrays."""
        X = np.array([1, 2, 3, 4, 5])
        detector = SOS()

        with pytest.raises(ValueError, match='Expected 2D array'):
            detector.fit(X)

    def test_fit_accepts_list(self) -> None:
        """Test fit accepts Python lists."""
        X = [[1, 2], [3, 4], [5, 6]]
        detector = SOS()
        detector.fit(X)

        assert detector.n_features_in_ == 2


class TestPredictValidation:
    """Test predict() validation behavior."""

    def test_predict_before_fit_raises(self, simple_data: np.ndarray) -> None:
        """Test predict before fit raises NotFittedError."""
        detector = SOS()

        with pytest.raises(NotFittedError):
            detector.predict(simple_data)

    def test_predict_wrong_n_features(self, simple_data: np.ndarray) -> None:
        """Test predict with wrong number of features raises."""
        detector = SOS()
        detector.fit(simple_data)  # 2 features

        X_wrong = np.array([[1, 2, 3], [4, 5, 6]])  # 3 features

        with pytest.raises(ValueError, match='Expected.*2.*features'):
            detector.predict(X_wrong)

    def test_predict_with_nan_raises(self, simple_data: np.ndarray) -> None:
        """Test predict with NaN raises."""
        detector = SOS()
        detector.fit(simple_data)

        X_nan = np.array([[1, 2], [3, np.nan]])

        with pytest.raises(ValueError, match='Input contains NaN'):
            detector.predict(X_nan)


class TestParameterValidation:
    """Test parameter validation."""

    def test_negative_perplexity_raises(self, simple_data: np.ndarray) -> None:
        """Test negative perplexity raises on fit."""
        detector = SOS(perplexity=-10)

        with pytest.raises(ValueError, match='perplexity.*positive'):
            detector.fit(simple_data)

    def test_zero_perplexity_raises(self, simple_data: np.ndarray) -> None:
        """Test zero perplexity raises on fit."""
        detector = SOS(perplexity=0)

        with pytest.raises(ValueError, match='perplexity.*positive'):
            detector.fit(simple_data)

    def test_negative_eps_raises(self, simple_data: np.ndarray) -> None:
        """Test negative eps raises on fit."""
        detector = SOS(eps=-1e-5)

        with pytest.raises(ValueError, match='eps.*positive'):
            detector.fit(simple_data)

    def test_invalid_metric_type_raises(self, simple_data: np.ndarray) -> None:
        """Test non-string metric raises on fit."""
        detector = SOS(metric=123)

        with pytest.raises(TypeError, match='metric.*string'):
            detector.fit(simple_data)


class TestConvenienceMethods:
    """Test convenience methods."""

    def test_fit_predict(self, simple_data: np.ndarray) -> None:
        """Test fit_predict returns same as fit().predict()."""
        detector1 = SOS(perplexity=10)
        scores1 = detector1.fit_predict(simple_data)

        detector2 = SOS(perplexity=10)
        detector2.fit(simple_data)
        scores2 = detector2.predict(simple_data)

        np.testing.assert_array_equal(scores1, scores2)

    def test_score_samples_equals_predict(self, simple_data: np.ndarray) -> None:
        """Test score_samples returns same as predict."""
        detector = SOS()
        detector.fit(simple_data)

        scores1 = detector.predict(simple_data)
        scores2 = detector.score_samples(simple_data)

        np.testing.assert_array_equal(scores1, scores2)

    def test_decision_function_equals_predict(self, simple_data: np.ndarray) -> None:
        """Test decision_function returns same as predict."""
        detector = SOS()
        detector.fit(simple_data)

        scores1 = detector.predict(simple_data)
        scores2 = detector.decision_function(simple_data)

        np.testing.assert_array_equal(scores1, scores2)
```

**Test Coverage Goals**:
- sklearn compatibility: 5 tests
- fit() validation: 7 tests
- predict() validation: 3 tests
- Parameter validation: 4 tests
- Convenience methods: 3 tests
- **Total: 22 new tests**

---

#### Step 3.2: Test pipeline integration

**File**: `tests/test_sklearn_integration.py` (add to same file)

```python
class TestPipelineIntegration:
    """Test integration with sklearn Pipeline."""

    def test_in_pipeline(self, simple_data: np.ndarray) -> None:
        """Test SOS works in sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('detector', SOS(perplexity=5)),
        ])

        pipe.fit(simple_data)
        scores = pipe.predict(simple_data)

        assert scores.shape == (len(simple_data),)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_pipeline_set_params(self, simple_data: np.ndarray) -> None:
        """Test set_params works through Pipeline."""
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ('detector', SOS()),
        ])

        pipe.set_params(detector__perplexity=15)
        assert pipe.named_steps['detector'].perplexity == 15

    def test_pipeline_get_params(self) -> None:
        """Test get_params works through Pipeline."""
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ('detector', SOS(perplexity=25)),
        ])

        params = pipe.get_params()
        assert params['detector__perplexity'] == 25
```

---

### Phase 4: Documentation (0.5 days)

#### Step 4.1: Update README.md

**File**: `README.md`

Add new section after "Usage":

```markdown
## scikit-learn Integration

As of version 0.3.0, SOS is a fully compliant scikit-learn estimator.

### Using in Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksos import SOS

# Create pipeline with preprocessing
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('outlier_detector', SOS(perplexity=20))
])

# Fit and predict
pipeline.fit(X)
scores = pipeline.predict(X)

# Update parameters
pipeline.set_params(outlier_detector__perplexity=30)
```

### API Methods

SOS implements the standard sklearn outlier detection interface:

```python
detector = SOS()

# Fit and predict separately
detector.fit(X)
scores = detector.predict(X)

# Or in one step
scores = detector.fit_predict(X)

# Alternative scoring methods (all equivalent)
scores = detector.score_samples(X)
scores = detector.decision_function(X)
```

### Input Validation

SOS validates input data and parameters:

```python
# Feature count must match between fit and predict
detector.fit(X_train)  # n_features = 10
detector.predict(X_test)  # Must also have 10 features

# Rejects invalid data
detector.fit(X_with_nan)  # ValueError: Input contains NaN
detector.fit(X_1d)  # ValueError: Expected 2D array
```

### Getting Parameters

```python
detector = SOS(perplexity=25, metric='manhattan')

# Get all parameters
params = detector.get_params()
# {'perplexity': 25, 'metric': 'manhattan', 'eps': 1e-5}

# Update parameters
detector.set_params(perplexity=30)
```
```

---

#### Step 4.2: Update API docstrings

Ensure all public methods have complete numpy-style docstrings:
- Parameters section with types
- Returns section
- Raises section for exceptions
- Examples section
- See Also section for related methods
- Notes section for algorithm details

---

#### Step 4.3: Add migration guide

**File**: `MIGRATION.md` (new file)

```markdown
# Migration Guide: v0.2.x to v0.3.0

## Overview

Version 0.3.0 adds full scikit-learn BaseEstimator integration. The API remains
backward compatible, but there are new features and recommended best practices.

## What's New

### New Methods

- `fit_predict(X)`: Convenience method for `fit(X).predict(X)`
- `score_samples(X)`: Alias for `predict(X)` (sklearn convention)
- `decision_function(X)`: Alias for `predict(X)` (sklearn convention)

### New Attributes

After calling `fit()`, the following attributes are available:

- `n_features_in_`: Number of features in training data
- `feature_names_in_`: Feature names (if input was DataFrame)

### Enhanced Validation

The `fit()` method now:
- Validates parameter values (perplexity > 0, eps > 0)
- Checks for NaN/Inf in input data
- Ensures input is 2D
- Stores feature metadata

The `predict()` method now:
- Requires `fit()` to be called first (raises NotFittedError)
- Validates number of features matches training data
- Checks for NaN/Inf in input data

## Breaking Changes

**None.** This release is fully backward compatible.

However, code that previously "worked" with invalid inputs will now raise
errors at appropriate times:

```python
# Before: Would fail deep in algorithm with cryptic error
detector = SOS(perplexity=-10)
detector.predict(X)  # Runtime error in binary search

# After: Fails immediately with clear error
detector = SOS(perplexity=-10)
detector.fit(X)  # ValueError: perplexity must be positive
```

## Recommended Updates

### Before (v0.2.x)

```python
from sksos import SOS

detector = SOS()
scores = detector.predict(X)  # fit() was optional no-op
```

### After (v0.3.0)

```python
from sksos import SOS

# Recommended: Call fit() explicitly
detector = SOS()
detector.fit(X_train)
scores = detector.predict(X_test)

# Or use fit_predict() for convenience
scores = detector.fit_predict(X)
```

## New Capabilities

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksos import SOS

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('detector', SOS(perplexity=20))
])

pipe.fit(X)
scores = pipe.predict(X)
```

### Parameter Inspection

```python
detector = SOS(perplexity=25)

# Get parameters
params = detector.get_params()

# Update parameters
detector.set_params(perplexity=30)

# Clone detector
from sklearn.base import clone
detector2 = clone(detector)
```

## Deprecations

None in this release.

## Future Deprecations

None planned. The API is considered stable.
```

---

### Phase 5: Update Existing Tests (0.5 days)

#### Step 5.1: Update existing test fixtures

**File**: `tests/conftest.py`

No changes needed, but verify compatibility.

---

#### Step 5.2: Update existing tests

**Files**: `tests/test_sos.py`, `tests/test_cli.py`

Review all tests to ensure they:
1. Call `fit()` before `predict()` (some may not currently)
2. Still pass with new validation
3. Don't make assumptions about internal behavior

Example changes needed:

```python
# Before
def test_predict_output_shape(self, simple_data):
    detector = SOS()
    O = detector.predict(simple_data)  # No fit()
    assert O.shape == (len(simple_data),)

# After
def test_predict_output_shape(self, simple_data):
    detector = SOS()
    detector.fit(simple_data)  # Add fit()
    O = detector.predict(simple_data)
    assert O.shape == (len(simple_data),)
```

Run tests and fix any that fail due to new validation.

---

## Code Changes Summary

### Files to Modify

1. **`pyproject.toml`**: Add scikit-learn dependency
2. **`sksos/sos.py`**: Major changes (inheritance, validation, new methods)
3. **`tests/test_sklearn_integration.py`**: New file (~300 LOC)
4. **`tests/test_sos.py`**: Minor updates (add fit() calls)
5. **`tests/test_cli.py`**: No changes needed
6. **`README.md`**: Add sklearn integration section
7. **`MIGRATION.md`**: New file

### Lines of Code Impact

- Production code: +~150 LOC (mostly docstrings and validation)
- Test code: +~300 LOC (new sklearn integration tests)
- Documentation: +~200 LOC (README + migration guide)
- **Total: ~650 LOC**

---

## Testing Strategy

### Test Execution Plan

1. **Run existing tests**: Ensure nothing breaks
   ```bash
   pytest tests/test_sos.py -v
   pytest tests/test_cli.py -v
   ```

2. **Run new sklearn tests**:
   ```bash
   pytest tests/test_sklearn_integration.py -v
   ```

3. **Run sklearn estimator checks**:
   ```bash
   pytest tests/test_sklearn_integration.py::test_estimator_checks -v
   ```
   Note: Some checks may not apply to outlier detectors. Document which
   checks are skipped and why.

4. **Test with different sklearn versions**:
   - sklearn 1.0.0 (minimum)
   - sklearn 1.2.0 (with parameter validation)
   - sklearn latest (currently 1.5+)

5. **Integration testing**:
   ```bash
   # Test in actual pipeline
   python -c "
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sksos import SOS
   import numpy as np

   X = np.random.randn(100, 5)
   pipe = Pipeline([('scaler', StandardScaler()), ('sos', SOS())])
   pipe.fit(X)
   print(pipe.predict(X))
   "
   ```

### Expected Test Coverage

After implementation:
- `sksos/sos.py`: ~95% (up from 97%, new validation code)
- `sksos/cli.py`: Still 0% (separate issue)
- Overall: ~80% (up from 74%)

---

## Documentation Updates

### README.md Updates

- Add "scikit-learn Integration" section
- Update examples to show fit() usage
- Add pipeline example
- Update "Type Hints" section to mention sklearn types

### New Documentation

- `MIGRATION.md`: Guide for upgrading from 0.2.x
- Docstrings: Complete numpy-style docs for all methods

### Future Documentation (not in this plan)

- Sphinx documentation site
- API reference
- User guide with tutorials
- Comparison with other outlier detectors

---

## Migration Guide

### For Users

**Backward Compatibility**: All existing code will continue to work.

**Recommended Changes**:
```python
# Old style (still works)
detector = SOS()
scores = detector.predict(X)

# New style (recommended)
detector = SOS()
detector.fit(X)
scores = detector.predict(X)

# Or
scores = detector.fit_predict(X)
```

### For Developers

When adding new parameters in the future:

1. Add to `__init__` with default value
2. Add to `_parameter_constraints` (if using sklearn 1.2+)
3. Add validation in `_validate_params_manual()`
4. Document in docstring
5. Add tests for validation

---

## Validation Checklist

Before considering this complete, verify:

### Code Quality
- [ ] All type hints present and correct
- [ ] All public methods have docstrings
- [ ] Docstrings follow numpy style
- [ ] mypy passes with no errors
- [ ] ruff check passes
- [ ] ruff format applied

### Testing
- [ ] All existing tests pass
- [ ] New sklearn integration tests pass
- [ ] Test coverage >= 80%
- [ ] Tests pass on Python 3.9, 3.10, 3.11, 3.12, 3.13
- [ ] Tests pass on Linux, macOS, Windows
- [ ] Tests pass with sklearn 1.0, 1.2, latest

### sklearn Compliance
- [ ] Inherits from BaseEstimator
- [ ] Inherits from OutlierMixin
- [ ] `fit()` validates input and stores metadata
- [ ] `predict()` checks is_fitted
- [ ] `get_params()` works (automatic from BaseEstimator)
- [ ] `set_params()` works (automatic from BaseEstimator)
- [ ] `clone()` works
- [ ] `repr()` is informative
- [ ] Works in sklearn Pipeline
- [ ] Parameters can be set via Pipeline.set_params()

### Documentation
- [ ] README updated with sklearn examples
- [ ] Migration guide created
- [ ] All methods documented
- [ ] Examples run without errors
- [ ] CHANGELOG.md updated

### API Consistency
- [ ] `fit_predict()` available
- [ ] `score_samples()` available
- [ ] `decision_function()` available
- [ ] All three return identical results
- [ ] Fitted attributes end with `_`
- [ ] n_features_in_ stored after fit()

---

## Risks & Mitigation

### Risk 1: Breaking Changes
**Risk**: New validation might break existing user code
**Likelihood**: Medium
**Impact**: High
**Mitigation**:
- Thorough testing with different input patterns
- Clear error messages
- Migration guide
- Consider deprecation warnings for problematic patterns

### Risk 2: sklearn Version Compatibility
**Risk**: Different sklearn versions behave differently
**Likelihood**: Low
**Impact**: Medium
**Mitigation**:
- Test with multiple sklearn versions
- Conditional imports for optional features
- Document minimum sklearn version clearly

### Risk 3: Performance Regression
**Risk**: Input validation adds overhead
**Likelihood**: Low
**Impact**: Low
**Mitigation**:
- Validation is only at fit/predict, not in inner loops
- sklearn's validation is optimized
- Benchmark before/after

### Risk 4: Incomplete sklearn Compatibility
**Risk**: Some sklearn features don't work as expected
**Likelihood**: Low
**Impact**: Medium
**Mitigation**:
- Run `check_estimator()` tests
- Document any known limitations
- Test common workflows (pipelines, etc.)

### Risk 5: Documentation Debt
**Risk**: New features not well documented
**Likelihood**: Medium
**Impact**: Medium
**Mitigation**:
- Complete docstrings for all new methods
- Update README with examples
- Create migration guide
- Add to CHANGELOG

---

## Timeline

### Day 1: Preparation & Core Integration
- Morning: Add sklearn dependency, study conventions (2 hours)
- Afternoon: Implement class inheritance and core methods (4 hours)
- Evening: Manual testing and fixes (2 hours)

### Day 2: Testing & Refinement
- Morning: Write sklearn integration tests (3 hours)
- Afternoon: Update existing tests, fix failures (3 hours)
- Evening: Run full test suite across environments (2 hours)

### Day 3: Documentation & Validation
- Morning: Update README, write migration guide (3 hours)
- Afternoon: Final testing, validation checklist (3 hours)
- Evening: Code review, documentation review (2 hours)

**Total: 24 hours (3 full days)**

---

## Success Criteria

This plan is successfully completed when:

1. ✅ SOS inherits from BaseEstimator and OutlierMixin
2. ✅ All sklearn integration tests pass
3. ✅ All existing tests still pass
4. ✅ `check_estimator()` passes (or known failures documented)
5. ✅ Works in sklearn Pipeline
6. ✅ Test coverage >= 80%
7. ✅ Documentation updated
8. ✅ Migration guide created
9. ✅ CI/CD pipeline passes on all platforms
10. ✅ Ready for release as v0.3.0

---

## Future Enhancements (Not in This Plan)

These are related but separate efforts:

1. **Hyperparameter tuning example**: Show GridSearchCV usage (requires scoring function)
2. **Model selection**: Compare with other outlier detectors
3. **Visualization**: Plot outlier scores, affinity matrix
4. **Serialization**: pickle/joblib support (should work automatically)
5. **Tags**: Add sklearn tags for estimator properties
6. **Sample weights**: Support sample_weight parameter (major change)

---

## Appendix A: sklearn Estimator Requirements

From sklearn documentation, an estimator should:

1. Inherit from BaseEstimator (provides get_params/set_params)
2. Store all constructor parameters as attributes
3. Not modify constructor parameters in __init__
4. Store fitted attributes with trailing `_`
5. Implement fit(X, y=None) that returns self
6. Validate input with _validate_data()
7. Check is fitted before predict()
8. Raise NotFittedError if not fitted
9. Have informative __repr__
10. Be clonable with sklearn.base.clone()

For outlier detectors specifically:
- Inherit from OutlierMixin
- Implement predict() returning outlier labels
- Optionally implement decision_function()
- Optionally implement score_samples()

**SOS will comply with all of these.**

---

## Appendix B: Code Diff Preview

High-level diff structure:

```diff
diff --git a/sksos/sos.py b/sksos/sos.py
--- a/sksos/sos.py
+++ b/sksos/sos.py
@@ -5,12 +5,18 @@
 import numpy as np
 from numpy.typing import ArrayLike, NDArray

+from sklearn.base import BaseEstimator, OutlierMixin
+from sklearn.utils.validation import check_is_fitted
+
-class SOS:
-    """Stochastic Outlier Selection."""
+class SOS(BaseEstimator, OutlierMixin):
+    """Stochastic Outlier Selection.
+
+    ... (extensive docstring) ...
+    """
+
+    _parameter_constraints = { ... }

-    def fit(self, X: ArrayLike) -> SOS:
-        """Fit the model (sklearn compatibility)."""
-        return self
+    def fit(self, X: ArrayLike, y=None) -> SOS:
+        """Validate data and store feature information."""
+        self._validate_params()
+        X = self._validate_data(X, ...)
+        return self

     def predict(self, X: ArrayLike) -> FloatArray:
         """Predict outlier scores."""
+        check_is_fitted(self, 'n_features_in_')
+        X = self._validate_data(X, reset=False)
         D = self.x2d(X)
         ...
+
+    def fit_predict(self, X, y=None):
+        """Fit and predict in one step."""
+        return self.fit(X, y).predict(X)
+
+    def score_samples(self, X):
+        """Compute outlier scores."""
+        return self.predict(X)
+
+    def decision_function(self, X):
+        """Compute decision function."""
+        return self.predict(X)
```

---

**End of Plan**

This plan is ready for review and implementation. Each step is detailed with rationale, code examples, and validation criteria.
