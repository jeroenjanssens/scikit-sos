Stochastic Outlier Selection
============================

Stochastic Outlier Selection (SOS) is an unsupervised outlier selection algorithm. It uses the concept of affinity to compute  an outlier probability for each data point.

![SOS](doc/sos.png)

For more information about SOS, see the technical report: J.H.M. Janssens, F. Huszar, E.O. Postma, and H.J. van den Herik. [Stochastic Outlier Selection](https://github.com/jeroenjanssens/sos/blob/master/doc/sos-ticc-tr-2012-001.pdf?raw=true). Technical Report TiCC TR 2012-001, Tilburg University, Tilburg, the Netherlands, 2012.

A Python implementation of the SOS algorithm can be found in the bin directory. This implementation only depends on NumPy and can be used from the command-line.

License
-------

All software in this repository is distributed under the terms of the BSD Simplified License. The full license is in the LICENSE file.
