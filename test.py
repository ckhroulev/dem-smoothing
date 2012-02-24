from leastsq import least_square_fit, bilinear_eval
import numpy as np

N = 51
x = np.linspace(-1,1,N)
y = x.copy()

xx, yy = np.meshgrid(x,y)
data = xx + yy + xx*yy + 0.5 * np.random.randn(xx.shape[0], xx.shape[1])

coeffs = least_square_fit(2.0/(N-1), 2.0/(N-1), data)

print "f(x,y) = %3.3f + %3.3f x + %3.3f y + %3.3f xy" % tuple(coeffs[0])

xxx = np.linspace(-1,1,101)
yyy = xxx.copy()

z = np.zeros((101,101))

for j in xrange(101):
    for i in xrange(101):
        z[j,i] = bilinear_eval(coeffs, xxx[i], yyy[j])

from pylab import *

pcolormesh(x,y,data)
colorbar()
grid(True)

cs = contour(xxx,yyy,z,21,colors='black', linewidths=2)
clabel(cs)

axes().set_aspect('equal')
show()
