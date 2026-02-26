# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-02-26

### Added

- `SOS` now inherits from `sklearn.base.BaseEstimator` and `sklearn.base.OutlierMixin`
- `fit()` method for input validation and feature metadata storage
- `fit_predict()` method for fitting and predicting in one call
- `score_samples()` method (alias for `predict()`)
- `decision_function()` method (alias for `predict()`)
- Automatic `get_params()` and `set_params()` from `BaseEstimator`
- Parameter validation using sklearn's `_parameter_constraints` (sklearn 1.2+) with manual fallback
- Input validation: rejects NaN, infinity, and 1D arrays
- `n_features_in_` attribute set during `fit()`
- `feature_names_in_` attribute set when fitting with pandas DataFrames
- Feature count validation in `predict()` (must match `fit()`)
- `NotFittedError` raised when calling `predict()` before `fit()`
- Pipeline and `clone()` compatibility
- Comprehensive numpy-style docstrings
- sklearn integration tests

### Changed

- `predict()` now requires `fit()` to be called first
- `scikit-learn>=1.0.0` is now a required dependency

### Dependencies

- Added `scikit-learn>=1.0.0`
