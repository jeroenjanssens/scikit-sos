scikit-sos
==========

scikit-sos is a Python module for Stochastic Outlier Selection (SOS). It
is compatible with scikit-learn.

SOS is an unsupervised outlier selection algorithm. It uses the concept
of affinity to compute an outlier probability for each data point.

.. figure:: doc/sos.png
   :alt: SOS

For more information about SOS, see the technical report: J.H.M.
Janssens, F. Huszar, E.O. Postma, and H.J. van den Herik. `Stochastic
Outlier
Selection <https://github.com/jeroenjanssens/sos/blob/master/doc/sos-ticc-tr-2012-001.pdf?raw=true>`__.
Technical Report TiCC TR 2012-001, Tilburg University, Tilburg, the
Netherlands, 2012.

Selecting outliers from the command line
----------------------------------------

A Python implementation of the SOS algorithm can be found in the sksos
directory. This implementation only depends on NumPy and can be used
from the command-line. For example, if we apply SOS with a perplexity of
20 to the Iris dataset, which is included, we obtain the following
outlier probabilities:

.. code:: bash

    git clone https://github.com/jeroenjanssens/sos.git 
    cd sos/sksos
    < iris.csv ./sos -p 30 | sort -nr | head
    0.92552418
    0.91794955
    0.81657372
    0.79410068
    0.77251273
    0.76652991
    0.71135211
    0.69634175
    0.69305280
    0.68967627

Adding a threshold causes SOS to output 0s and 1s instead of outlier
probabilities. If we set the threshold to 0.75 then we see that out of
the 150 data points, 6 are selected as outliers:

.. code:: bash

    < iris.csv ./sos -p 30 -t 0.75 | paste -sd+ | bc
    6

Under the hood, SOS simply needs a two-dimensional NumPy array. A PyPI
package is in the making.

License
-------

All software in this repository is distributed under the terms of the
BSD Simplified License. The full license is in the LICENSE file.
