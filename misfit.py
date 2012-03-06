from netCDF4 import Dataset as NC
import numpy as np

nc_dem = NC("data/greenland_5km.nc", 'r')
nc_data = NC("data/greenland_5km_smooth.nc", 'r')

# Get data
thk = np.squeeze(nc_dem.variables['thk'])
usurf = np.squeeze(nc_dem.variables['usurf'])
x = nc_dem.variables['x'][:]
y = nc_dem.variables['y'][:]

vel_mag = np.squeeze(nc_dem.variables['surfvelmag'])
vel_x   = np.squeeze(nc_dem.variables['surfvelx'])
vel_y   = np.squeeze(nc_dem.variables['surfvely'])

coeffs = nc_data.variables['data']

def compute_misfit(v1, v2, tol):
    """Given 2 2D vectors v1 and v2 computes the misfit function
    f(v1,v2) = 1 - cos(theta), where theta is the angle between vectors.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if (n1 > tol) and (n2 > tol):
        return np.rad2deg(np.arccos(np.vdot(v1, v2) / (n1 * n2)))

    return 180

x_size = thk.shape[1]
y_size = thk.shape[0]
n_levels = coeffs.shape[2]

misfit   = np.zeros((thk.shape[0], thk.shape[1], n_levels))

for j in xrange(y_size):
    for i in xrange(x_size):
        # skip areas without ice
        if thk[j,i] < 10:
            misfit[j,i] = -1
            continue

        # skip areas with very little flow
        if vel_mag[j,i] < 10:
            misfit[j,i] = -1
            continue

        v1 = [vel_x[j,i], vel_y[j,i]]

        cs = coeffs[j,i]

        for k in xrange(n_levels):
            v2 = [-cs[k][1], -cs[k][2]] # flow *down* the gradient
            misfit[j,i,k] = compute_misfit(v1, v2, 1e-5)

import PISMNC

nc = PISMNC.PISMDataset("misfit.nc", 'w')

nc.create_dimensions(x,y)
nc.createDimension("level", n_levels)
data = nc.createVariable('data', 'f8', ('y', 'x', 'level'))

data[:] = misfit

nc.close()



