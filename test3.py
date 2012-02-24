from leastsq import least_square_sequence, bilinear_eval
import numpy as np

N = 101
x = np.linspace(-1,1,N)
y = x.copy()

xx, yy = np.meshgrid(x,y)
data = xx + yy + xx*yy + 0.5 * np.random.randn(xx.shape[0], xx.shape[1])

coeffs = least_square_sequence(2.0/(N-1), 2.0/(N-1), data)

for C in coeffs:
    print "f(x, y) = %7.3f + %7.3f x + %6.3f y + %8.3f xy" % tuple(C[0])

z = np.zeros_like(xx)

from pylab import *

C = coeffs[0]
for j in xrange(N):
    for i in xrange(N):
        z[j,i] = bilinear_eval(C, x[i], y[j])

figure(N)
pcolormesh(x, y, data)
grid(True)
hold(True)
cs = contour(xx, yy, z, 21, colors='black', linewidths=2)
clabel(cs)

axes().set_aspect('equal')

show()
