# scikit-sos

scikit-sos is a Python module for Stochastic Outlier Selection (SOS). It is compatible with scikit-learn. SOS is an unsupervised outlier selection algorithm. It uses the concept of affinity to compute an outlier probability for each data point.

![SOS](https://github.com/jeroenjanssens/scikit-sos/raw/master/doc/sos.png)

For more information about SOS, see the technical report: J.H.M. Janssens, F. Huszar, E.O. Postma, and H.J. van den Herik. [Stochastic Outlier Selection](https://github.com/jeroenjanssens/sos/blob/master/doc/sos-ticc-tr-2012-001.pdf?raw=true). Technical Report TiCC TR 2012-001, Tilburg University, Tilburg, the Netherlands, 2012.

## Install

Using pip:

```bash
pip install scikit-sos
```

Using uv (recommended for fast installation):

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install scikit-sos
uv pip install scikit-sos
```

## Development

This project uses modern Python tooling:

- **uv** for fast package management
- **ruff** for linting and formatting
- **mypy** for type checking
- **pytest** for testing

To set up a development environment:

```bash
# Clone repository
git clone https://github.com/jeroenjanssens/scikit-sos.git
cd scikit-sos

# Create virtual environment and install with dev dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

Run formatting:

```bash
ruff format .
```

Run type checking:

```bash
mypy sksos
```

### Type Hints

This package includes full type hints for better IDE support:

```python
from sksos import SOS
import numpy as np
from numpy.typing import NDArray

# Type hints work automatically
detector: SOS = SOS(perplexity=20)
data: NDArray = np.array([[1, 2], [3, 4]])
scores: NDArray = detector.fit(data).predict(data)
```

## Usage

```python
>>> import pandas as pd
>>> from sksos import SOS
>>> iris = pd.read_csv("http://bit.ly/iris-csv")
>>> X = iris.drop("Name", axis=1).values
>>> detector = SOS()
>>> iris["score"] = detector.fit(X).predict(X)
>>> iris.sort_values("score", ascending=False).head(10)
     SepalLength  SepalWidth  PetalLength  PetalWidth             Name     score
41           4.5         2.3          1.3         0.3      Iris-setosa  0.981898
106          4.9         2.5          4.5         1.7   Iris-virginica  0.964381
22           4.6         3.6          1.0         0.2      Iris-setosa  0.957945
134          6.1         2.6          5.6         1.4   Iris-virginica  0.897970
24           4.8         3.4          1.9         0.2      Iris-setosa  0.871733
114          5.8         2.8          5.1         2.4   Iris-virginica  0.831610
62           6.0         2.2          4.0         1.0  Iris-versicolor  0.821141
108          6.7         2.5          5.8         1.8   Iris-virginica  0.819842
44           5.1         3.8          1.9         0.4      Iris-setosa  0.773301
100          6.3         3.3          6.0         2.5   Iris-virginica  0.765657
```

## scikit-learn Integration

SOS is a fully compatible scikit-learn estimator, inheriting from `BaseEstimator` and `OutlierMixin`.

### Using in Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksos import SOS

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('sos', SOS(perplexity=20)),
])
pipe.fit(X)
scores = pipe.predict(X)
```

### API Methods

```python
detector = SOS(perplexity=20)

# Fit and predict separately
detector.fit(X)
scores = detector.predict(X)

# Or in one call
scores = detector.fit_predict(X)

# Aliases
scores = detector.score_samples(X)
scores = detector.decision_function(X)
```

### Parameter Inspection

```python
detector = SOS(perplexity=20, metric='euclidean')
detector.get_params()   # {'perplexity': 20, 'metric': 'euclidean', 'eps': 1e-05}
detector.set_params(perplexity=30)
```

## Command Line Interface

This module also includes a command-line tool called `sos`. To illustrate, we apply SOS with a perplexity of 10 to the Iris dataset:

```bash
$ curl -sL http://bit.ly/iris-csv |
> tail -n +2 | cut -d, -f1-4 |
> sos -p 10 |
> sort -nr | head
0.98189840
0.96438132
0.95794492
0.89797043
0.87173299
0.83161045
0.82114072
0.81984209
0.77330148
0.76565738
```

Adding a threshold causes SOS to output 0s and 1s instead of outlier probabilities. If we set the threshold to 0.8 then we see that out of the 150 data points, 8 are selected as outliers:

```bash
$ curl -sL http://bit.ly/iris-csv |
> tail -n +2 | cut -d, -f1-4 |
> sos -p 10 -t 0.8 |
> paste -sd+ | bc
8
```

## License

All software in this repository is distributed under the terms of the BSD Simplified License. The full license is in the LICENSE file.
