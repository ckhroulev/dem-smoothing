import numpy as np
from leastsq import least_square_sequence
import sys

try:
    from netCDF4 import Dataset as NC
except:
    from netCDF3 import Dataset as NC

nc = NC(sys.argv[1])

x = nc.variables['x'][:]
y = nc.variables['y'][:]

dx = x[1] - x[0]
dy = y[1] = y[0]

dem = nc.variables['usurf'][:]

nc.close()

# max window width:
N = 11

# number of smoothing levels:
M = N

# number of coefficients (bilinear approximation)
P = 4

ii, jj = np.meshgrid(np.r_[-N:N+1],
                     np.r_[-N:N+1])
dem_data = np.zeros((dem.shape[0], dem.shape[1], M, P))

for j0 in xrange(N, y.size - N):
    for i0 in xrange(N, x.size - N):
        i = i0 + ii
        j = j0 + jj

        data = dem[j,i]

        dem_data[j0,i0] = np.array(least_square_sequence(dx, dy, data)).squeeze()

nc = NC("output.nc", 'w')

nc.createDimension('x', x.size)
nc.createDimension('y', y.size)
nc.createDimension('level', M)
nc.createDimension('coeff', P)

data = nc.createVariable('data', 'f8', ('y', 'x', 'level', 'coeff'))

data[:] = dem_data

nc.close()
