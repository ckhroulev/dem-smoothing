import numpy as np
import leastsq
import sys

try:
    from netCDF4 import Dataset as NC
except:
    from netCDF3 import Dataset as NC

nc = NC(sys.argv[1])

x = nc.variables['x'][:]
y = nc.variables['y'][:]
dem = np.squeeze(nc.variables['usurf'][:])

nc.close()

# x and y better be strictly increasing
dx = x[1] - x[0]
dy = y[1] - y[0]

nc = NC("output.nc", 'w')

n_levels = 10
nc.createDimension('x', x.size)
nc.createDimension('y', y.size)
nc.createDimension('level', n_levels)
nc.createDimension('coeff', leastsq.n_params)

data = nc.createVariable('data', 'f8', ('y', 'x', 'level', 'coeff'))

data[:] = leastsq.smooth(dem, dx, dy, n_levels)

nc.close()
