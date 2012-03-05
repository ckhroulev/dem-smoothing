# Time 10000 executions:
setup = """
from leastsq import least_square_sequence
import numpy as np
N = 15
x = np.linspace(-1,1,N)
y = x.copy()
dx = dy = 2.0/(N-1)
xx, yy = np.meshgrid(x,y)
data = xx + yy + xx*yy + 0.5 * np.random.randn(xx.shape[0], xx.shape[1])
"""
from timeit import Timer
timer = Timer(stmt = "coeffs = least_square_sequence(dx, dy, data)",
              setup = setup)

print timer.repeat(repeat=3, number = int(1e5))

from leastsq import least_square_sequence, bilinear_eval
import numpy as np
N = 15
x = np.linspace(-1,1,N)
y = x.copy()
dx = dy = 2.0/(N-1)
xx, yy = np.meshgrid(x,y)
data = xx + yy + xx*yy + 0.5 * np.random.randn(xx.shape[0], xx.shape[1])

C = least_square_sequence(dx, dy, data)

for coeffs in C:
    print "f(x,y) = %3.3f + %3.3f x + %3.3f y + %3.3f xy" % tuple(coeffs[0])

xxx = np.linspace(-1,1,101)
yyy = xxx.copy()

z = np.zeros((101,101))

for j in xrange(101):
    for i in xrange(101):
        z[j,i] = bilinear_eval(C[0], xxx[i], yyy[j])

from pylab import *

pcolormesh(x,y,data)
colorbar()
grid(True)

cs = contour(xxx,yyy,z,21,colors='black', linewidths=2)
clabel(cs)

axes().set_aspect('equal')
show()

