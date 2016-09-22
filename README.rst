scikit-sos
==========

scikit-sos is a Python module for Stochastic Outlier Selection (SOS). It
is compatible with scikit-learn. SOS is an unsupervised outlier selection
algorithm. It uses the concept of affinity to compute an outlier probability
for each data point.

.. figure:: https://github.com/jeroenjanssens/scikit-sos/raw/master/doc/sos.png
   :alt: SOS

For more information about SOS, see the technical report: J.H.M.
Janssens, F. Huszar, E.O. Postma, and H.J. van den Herik. `Stochastic
Outlier
Selection <https://github.com/jeroenjanssens/sos/blob/master/doc/sos-ticc-tr-2012-001.pdf?raw=true>`__.
Technical Report TiCC TR 2012-001, Tilburg University, Tilburg, the
Netherlands, 2012.

Install
-------

.. code:: bash

   pip install scikit-sos


Usage
-----

.. code:: python

    >>> import pandas as pd
    >>> from sksos import SOS
    >>> iris = pd.read_csv("http://bit.ly/iris-csv")
    >>> X = iris.drop("Name", axis=1).values
    >>> iris["score"] = detector.predict(X)
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


Selecting outliers from the command line
----------------------------------------

This module also includes a command-line tool called `sos`.
To illustrate, we apply SOS with a perplexity of 10 to the Iris dataset:

.. code:: bash

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


Adding a threshold causes SOS to output 0s and 1s instead of outlier
probabilities. If we set the threshold to 0.8 then we see that out of
the 150 data points, 8 are selected as outliers:

.. code:: bash

    $ curl -sL http://bit.ly/iris-csv |
    > tail -n +2 | cut -d, -f1-4 |
    > sos -p 10 -t 0.8 |
    > paste -sd+ | bc
    8


License
-------

All software in this repository is distributed under the terms of the
BSD Simplified License. The full license is in the LICENSE file.
