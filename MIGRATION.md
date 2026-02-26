# Migration Guide: v0.2.x to v0.3.0

## Overview

v0.3.0 makes `SOS` a fully compliant scikit-learn estimator by inheriting from `BaseEstimator` and `OutlierMixin`. This is a non-breaking change for most users, but the recommended usage pattern now includes an explicit `fit()` call.

## What's New

### New Methods

- `fit(X)` - validates input and stores feature metadata
- `fit_predict(X)` - fits and predicts in one call
- `score_samples(X)` - alias for `predict()`
- `decision_function(X)` - alias for `predict()`
- `get_params()` / `set_params()` - inherited from `BaseEstimator`

### New Attributes (after fitting)

- `n_features_in_` - number of features seen during fit
- `feature_names_in_` - feature names when fitting with pandas DataFrames

### Enhanced Validation

- Input data is validated for NaN, infinity, and correct shape
- Parameters are validated when `fit()` is called
- `predict()` raises `NotFittedError` if called before `fit()`
- `predict()` validates that input has the same number of features as training data

## Recommended Updates

### Before (v0.2.x)

```python
detector = SOS()
scores = detector.predict(X)
```

### After (v0.3.0)

```python
detector = SOS()
scores = detector.fit(X).predict(X)

# Or equivalently:
scores = detector.fit_predict(X)
```

## New Capabilities

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('sos', SOS(perplexity=20)),
])
scores = pipe.fit(X).predict(X)
```

### Parameter Inspection and Cloning

```python
from sklearn.base import clone

detector = SOS(perplexity=15)
detector.get_params()  # {'perplexity': 15, 'metric': 'euclidean', 'eps': 1e-05}

cloned = clone(detector)  # Fresh copy with same parameters
```
