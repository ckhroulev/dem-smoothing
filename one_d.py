#!/usr/bin/env python
import numpy as np
from sys import stderr

def vandermonde_quadratic_row(x):
    """
    Evaluates a row of the Vandermonde matrix
    corresponding to the point x and the quadratic function.
    """
    return [1, x, x*x]

def quadratic_eval(coeffs, x):
    """
    Evaluate the quadratic function given its coefficients and a point.

    Note: does not support vector arguments (x).
    """
    return np.vdot(coeffs, [1, x, x*x])

def vandermonde(row_func, xx):
    """
    Builds the Vandermonde matrix corresponding to points xx.
    """
    result = []
    for x in xx:
        result.append(row_func(x))

    return np.matrix(result)

def vandermonde_neighborhood(dx, N):
    """
    Builds a Vandermonde matrix for a quadratic least-squares approximation
    in the neighborhood of a cell of a regular grid.

    - dx -- grid spacing
    - N -- the number of cells around the current one to include.
      N = 1 would use immediate neighbors only, i.e. 3 points.
    """
    xx = np.linspace(-N*dx, N*dx, 2*N + 1)

    return vandermonde(vandermonde_quadratic_row, xx)

dictionary = {}

def least_square_fit(dx, data):
    """
    Returns coefficients of the quadratic least-squares approximation of 'data',
    assuming that 'data' is given on a regular grid with spacing dx,dy.

    Use quadratic_eval to evaluate it.
    """
    if data.ndim != 1:
        raise ValueError("data has to be a 1D array")

    N = data.shape[0]

    if N % 2 != 1:
        raise ValueError("data has to have an odd number of rows")

    if N < 3:
        raise ValueError("data has to have at least 3 rows, got %d" % N)

    i = (N - 1) / 2

    try:
        u, s, v = dictionary[i]
    except:
        u, s, v = np.linalg.svd(vandermonde_neighborhood(dx, i), full_matrices = False)
        dictionary[i] = u, s, v

    w = np.dot(data.flatten(), u)
    w /= s

    return np.asarray(w * v)

def least_square_sequence(dx, data):
    """
    Builds a sequence of quadratic least-squares approximations to decreasing subsets of 'data'.

    Here 'data' is thought of as the neighborhood of the center cell 'data'.
    We build approximations of 'data' at its center using smaller and smaller neighborhoods.

    Returns a list of coefficients, the using all of 'data' first, then data[1:N-1], etc.
    """
    N = data.shape[0]

    if N % 2 != 1:
        raise ValueError("data has to have an odd number of rows")

    result = []

    while data.shape[0] > 2:
        result.append(least_square_fit(dx, data))
        data = data[1:-1]
        stderr.write(".")

    stderr.write("\n")

    return result
