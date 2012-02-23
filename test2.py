import numpy as np

from leastsq import least_square_fit, bilinear_eval

try:
    from netCDF4 import Dataset as NC
except:
    from netCDF3 import Dataset as NC

nc = NC("/Users/constantine/drainagebasin/data/searise-greenland.nc")

x = nc.variables['x'][:]
y = nc.variables['y'][:]

dx = x[1] - x[0]
dy = y[1] = y[0]

dem = nc.variables['usurf'][:]

N = 1

i = np.r_[-N:N+1]
j = i.copy()

ii,jj = np.meshgrid(i,j)

dem_out = np.zeros_like(dem)

for j0 in xrange(N, y.size - N):
    for i0 in xrange(N, x.size - N):
        i = i0 + ii
        j = j0 + jj

        data = np.array(dem[j,i], order='C')

        coeffs = least_square_fit(dx, dy, data)

        dem_out[j,i] = np.sqrt(coeffs[0][1]**2 + coeffs[0][2]**2)


from pylab import *

figure(1)
pcolormesh(x, y, dem)
colorbar()
axis('tight')
axes().set_aspect('equal')

figure(2)
pcolormesh(x, y, dem_out)
colorbar()
axis('tight')
axes().set_aspect('equal')

show()
