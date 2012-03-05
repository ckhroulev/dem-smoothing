from PISMNC import PISMDataset as NC
import numpy as np

nc = NC("rings.nc", 'w')

N = 201
x = np.linspace(-2, 2, N)
y = x

dem = np.zeros((N,N))

for j in xrange(N):
    for i in xrange(N):
        x0, y0 = x[i], y[j]
        r = np.sqrt(x0*x0 + y0*y0)
        if np.abs(r / 0.5 - np.floor(r / 0.5)) < 0.45 and r <= 1.5:
            dem[j,i] = 1

nc.create_dimensions(x,y)
nc.write_2d_field("usurf", dem)

nc.close()
