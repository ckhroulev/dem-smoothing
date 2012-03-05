setup = """
from leastsq import least_square_sequence
import numpy as np
N = 101
x = np.linspace(-1,1,N)
y = x.copy()
dx = dy = 2.0/(N-1)
xx, yy = np.meshgrid(x,y)
data = xx + yy + xx*yy + 0.5 * np.random.randn(xx.shape[0], xx.shape[1])
least_square_sequence(dx, dy, data)"""

from timeit import Timer
timer = Timer(stmt = "coeffs = least_square_sequence(dx, dy, data)",
              setup = setup)

print timer.repeat(repeat=3, number = int(1e3))
