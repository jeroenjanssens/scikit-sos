import numpy as np

class SOS(object):

    """Stochastic Outlier Selection.

    Copyright (c) 2013, Jeroen Janssens
    All rights reserved.

    Distributed under the terms of the BSD Simplified License.
    The full license is in the LICENSE file, distributed with this software.

    For more information about SOS, see https://github.com/jeroenjanssens/sos
    J.H.M. Janssens, F. Huszar, E.O. Postma, and H.J. van den Herik. Stochastic
    Outlier Selection. Technical Report TiCC TR 2012-001, Tilburg University,
    Tilburg, the Netherlands, 2012.

    Please note that because SOS is inspired by t-SNE (created by Laurens
    van der Maaten; see http://homepage.tudelft.nl/19j49/t-SNE.html),
    this code borrows functionality from the Python implementation,
    namely the functions x2p and Hbeta.

    """

    def __init__(self, perplexity=30, metric='euclidean', eps=1e-5):
        self.perplexity = perplexity
        self.metric = metric.lower()
        self.eps = eps


    def x2d(self, X):
        """Computer dissimilarity matrix."""

        (n, d) = X.shape
        if self.metric == 'none':
            if n != d:
                raise ValueError("If you specify 'none' as the metric, the data set "
                    "should be a square dissimilarity matrix")
            else:
                D = X
        elif self.metric == 'euclidean':
            sumX = np.sum(np.square(X), 1)

            # np.abs protects against extremely small negative values
            # that may arise due to floating point arithmetic errors
            D = np.sqrt( np.abs(np.add(np.add(-2 * np.dot(X, X.T), sumX).T, sumX)) )
        else:
            try:
                from scipy.spatial import distance
            except ImportError as e:
                raise ImportError("Please install scipy if you wish to use a metric "
                    "other than 'euclidean' or 'none'")
            else:
                D = distance.squareform(distance.pdist(X, self.metric))
        return D


    def d2a(self, D):
        """Return affinity matrix.

        Performs a binary search to get affinities in such a way that each
        conditional Gaussian has the same perplexity.

        """

        (n, _) = D.shape
        A = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(self.perplexity)

        for i in range(n):
            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf
            betamax =  np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            (H, thisA) = get_perplexity(Di, beta[i])

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while np.isnan(Hdiff) or (np.abs(Hdiff) > self.eps and tries < 5000):
                if np.isnan(Hdiff):
                    beta[i] = beta[i] / 10.0
                # If not, increase or decrease precision
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
                # Recompute the values
                (H, thisA) = get_perplexity(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # Set the final row of A
            A[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisA

        return A

    def a2b(self, A):
        B = A / A.sum(axis=1)[:,np.newaxis]
        return B

    def b2o(self, B):
        O = np.prod(1-B, 0)
        return O

    def fit(self, X):
        pass

    def predict(self, X):
        D = self.x2d(X)
        A = self.d2a(D)
        B = self.a2b(A)
        O = self.b2o(B)
        return O


def get_perplexity(D, beta):
    """Compute the perplexity and the A-row for a specific value of the
    precision of a Gaussian distribution.

    """

    A = np.exp(-D * beta)
    sumA = sum(A)
    H = np.log(sumA) + beta * np.sum(D * A) / sumA
    return H, A

