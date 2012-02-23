#!/usr/bin/env python

import numpy as np

def vandermonde_bilinear_row(x, y):
    """
    Evaluates a row of the Vandermonde matrix corresponding to the point x,y and the bilinear function.
    """
    return [1, x, y, x*y]

def bilinear_eval(coeffs, x, y):
    """Evaluate the bilinear function given its coefficients and a point."""

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
    x = np.linspace(-N*dx, N*dx, 2*N + 1)
    y = np.linspace(-N*dy, N*dy, 2*N + 1)

    xx, yy = np.meshgrid(x,y)

    return vandermonde(vandermonde_bilinear_row, xx.flat, yy.flat)

dictionary = {}

def least_square_fit(dx, dy, data):
    if data.ndim != 2:
        raise ValueError("data has to be a 2D array")

    N = data.shape[0]

    if N != data.shape[1]:
        raise ValueError("data has to be square")

    if N % 2 != 1:
        raise ValueError("data has to have an odd number of rows")

    i = (N - 1) / 2

    try:
        u, s, v = dictionary[i]
    except:
        dictionary[i] = np.linalg.svd(vandermonde_neighborhood(dx, dy, i), full_matrices = False)
        u, s, v = dictionary[i]

    w = np.dot(u.H, data.flatten()) / s

    x = v.H * w.H

    return np.array(x.H)


N = 51
x = np.linspace(-1,1,N)
y = x.copy()

xx, yy = np.meshgrid(x,y)
data = xx + yy + xx*yy + np.random.randn(xx.shape[0], xx.shape[1])

coeffs = least_square_fit(2.0/(N-1), 2.0/(N-1), data)

xx = np.linspace(-1,1,101)
yy = xx.copy()

z = np.zeros((101,101))

for j in xrange(101):
    for i in xrange(101):
        z[j,i] = bilinear_eval(coeffs, xx[i], yy[j])

from pylab import *

pcolormesh(x,y,data)
colorbar()

cs = contour(xx,yy,z,10,colors='black', linestyles='solid', linewidths=2)
clabel(cs)

axes().set_aspect('equal')
show()
