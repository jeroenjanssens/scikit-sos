#!/usr/bin/env python

import argparse
import logging
import sys

import numpy as np

from sksos import SOS

def get_stdout():
    if sys.version_info.major < 3:
        return sys.stdout
    else:
        return sys.stdout.buffer


def main():
    parser = argparse.ArgumentParser(description="Stochastic Outlier Selection")
    parser.add_argument('-t', '--threshold', type=float, default=None,
        help=("Float between 0.0 and 1.0 to use as threshold for selecting "
            "outliers. By default, this is not set, causing the outlier "
            "probabilities instead of the classification to be outputted"))
    parser.add_argument('-d', '--delimiter', type=str, default=',', help=(
        "String to use to separate values. By default, this is a comma."))
    parser.add_argument('-i', '--input', type=argparse.FileType('rb'),
        default=sys.stdin, help=("File to read data set from. By default, "
            "this is <stdin>."))
    parser.add_argument('-m', '--metric', type=str, default='euclidean', help=(
        "String indicating the metric to use to compute the dissimilarity "
        "matrix. By default, this is 'euclidean'. Use 'none' if the data set "
        "is a dissimilarity matrix."))
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'),
        default=get_stdout(), help=("File to write the computed outlier "
            "probabilities to. By default, this is <stdout>."))
    parser.add_argument('-p', '--perplexity', type=float, default=30.0,
        help="Float to use as perpexity. By default, this is 30.0.")
    args = parser.parse_args()

    X = np.loadtxt(args.input, delimiter=args.delimiter, ndmin=2)

    O = SOS(args.perplexity, args.metric).predict(X)

    if args.threshold is None:
        np.savetxt(args.output, O, '%1.8f')
    else:
        np.savetxt(args.output, O>=args.threshold, '%1d')

    return 0

if __name__ == '__main__':
    exit(main())

