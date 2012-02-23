#!/usr/bin/env python

import numpy as np

def vandermonde_bilinear_row(x, y):
    """
    Evaluates a row of the Vandermonde matrix corresponding to the point x,y and the bilinear function.
    """
    return [1, x, y, x*y]

def vandermonde_bilinear(row_func, xx, yy):
    """
    Builds the Vandermonde matrix corresponding to points xx,yy.
    """

    result = []
    for (x,y) in zip(xx,yy):
        result.append(row_func(x,y))

    return np.matrix(result)

