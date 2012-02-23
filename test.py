from leastsq import least_square_fit, bilinear_eval
import numpy as np

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
