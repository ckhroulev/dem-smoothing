#!/usr/bin/env python
import numpy as np

dictionary = {}
n_params = 5                            # bilinear + gradient magnitude

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

def clear_precomputed_svds():
    global dictionary
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

    return (w * v).tolist()

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

    return result

def smooth(dem, dx, dy, n_levels):
    """
    Smooths the 'dem' on a regular grid with spacing 'dx' and 'dy', using
    n_levels levels of smoothing.
    """
    # reset the dictionary containing precomputed SVDs
    clear_precomputed_svds()

    # number of smoothing levels (and the half-width of the window):
    M = n_levels
    x_size = dem.shape[1]
    y_size = dem.shape[0]

    ii, jj = np.meshgrid(np.r_[-M:M+1],
                         np.r_[-M:M+1])
    output_data = np.zeros((dem.shape[0], dem.shape[1], M, n_params))

    for j0 in xrange(M, y_size - M):
        for i0 in xrange(M, x_size - M):
            i = i0 + ii
            j = j0 + jj

            try:
                data = dem[j,i].filled(0)
            except:
                data = dem[j,i]

            coeffs = least_square_sequence(dx, dy, data)

            result = []
            for c in coeffs:
                tmp = c[0]
                tmp.append(np.sqrt(tmp[1]*tmp[1] + tmp[2]*tmp[2]))
                result.append(tmp)

            output_data[j0,i0] = result

    # reset the dictionary containing precomputed SVDs
    clear_precomputed_svds()

    return output_data
