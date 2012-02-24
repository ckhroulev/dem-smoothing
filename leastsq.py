#!/usr/bin/env python
import numpy as np
from sys import stderr

def vandermonde_bilinear_row(x, y):
    """
    Evaluates a row of the Vandermonde matrix
    corresponding to the point x,y and the bilinear function.
    """
    return [1, x, y, x*y]

def bilinear_eval(coeffs, x, y):
    """
    Evaluate the bilinear function given its coefficients and a point.

    Note: does not support vector arguments (x and y).
    """
    return np.vdot(coeffs, [1, x, y, x*y])

def vandermonde(row_func, xx, yy):
    """
    Builds the Vandermonde matrix corresponding to points xx,yy.
    """
    result = []
    for (x,y) in zip(xx,yy):
        result.append(row_func(x,y))

    return np.matrix(result)

def vandermonde_neighborhood(dx, dy, N):
    """
    Builds a Vandermonde matrix for a bilinear least-squares approximation
    in the neighborhood of a cell of a regular grid.

    - dx, dy -- grid spacing
    - N -- the number of cells around the current one (in x and y directions) to include.
      N = 1 would use immediate neighbors only, i.e. using a 3 x 3 square.
    """
    x = np.linspace(-N*dx, N*dx, 2*N + 1)
    y = np.linspace(-N*dy, N*dy, 2*N + 1)

    xx, yy = np.meshgrid(x,y)

    return vandermonde(vandermonde_bilinear_row, xx.flat, yy.flat)

dictionary = {}

def least_square_fit(dx, dy, data):
    """
    Returns coefficients of the bilinear least-squares approximation of 'data',
    assuming that 'data' is given on a regular grid with spacing dx,dy.

    Use bilinear_eval to evaluate it.
    """
    if data.ndim != 2:
        raise ValueError("data has to be a 2D array")

    N = data.shape[0]

    if N != data.shape[1]:
        raise ValueError("data has to be square")

    if N % 2 != 1:
        raise ValueError("data has to have an odd number of rows")

    if N < 3:
        raise ValueError("data has to have at least 3 rows, got %d" % N)

    i = (N - 1) / 2

    try:
        u, s, v = dictionary[i]
    except:
        u, s, v = np.linalg.svd(vandermonde_neighborhood(dx, dy, i), full_matrices = False)
        dictionary[i] = u, s, v

    w = np.dot(data.flatten(), u)
    w /= s

    return np.asarray(w * v)

def least_square_sequence(dx, dy, data):
    """
    Builds a sequence of bilinear least-squares approximations to decreasing subsets of 'data'.

    Here 'data' is thought of as the neighborhood of the center cell 'data'.
    We build approximations of 'data' at its center using smaller and smaller neighborhoods.

    Returns a list of coefficients, the using all of 'data' first, then data[1:N-1,1:N-1], etc.
    """
    N = data.shape[0]

    if N != data.shape[1]:
        raise ValueError("data has to be square")

    if N % 2 != 1:
        raise ValueError("data has to have an odd number of rows")

    result = []

    while data.shape[0] > 2:
        result.append(least_square_fit(dx, dy, data))
        data = data[1:-1, 1:-1]
        stderr.write(".")

    stderr.write("\n")

    return result
