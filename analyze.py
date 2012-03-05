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
        return 1 - np.vdot(v1, v2) / (n1 * n2)

    return 2

x_size = thk.shape[1]
y_size = thk.shape[0]
n_levels = coeffs.shape[2]

scale = np.zeros_like(thk)
misfit   = np.zeros_like(thk)

for j in xrange(y_size):
    for i in xrange(x_size):
        # skip areas without ice
        if thk[j,i] < 10:
            scale[j,i] = -1
            misfit[j,i] = -1
            continue

        # skip areas with very little flow
        if vel_mag[j,i] < 10:
            scale[j,i] = -1
            misfit[j,i] = -1
            continue

        v1 = [vel_x[j,i], vel_y[j,i]]

        cs = coeffs[j,i]

        min_misfit = 3
        index = 0
        for k in xrange(n_levels - 1, -1, -1):
            v2 = [-cs[k][1], -cs[k][2]] # flow *down* the gradient
            err = compute_misfit(v1, v2, 1e-5)

            if err < min_misfit:
                index = k
                min_misfit = err

        scale[j,i] = n_levels - index
        misfit[j,i]   = err

import PISMNC

nc = PISMNC.PISMDataset("scales.nc", 'w')

nc.create_dimensions(x,y)
nc.write_2d_field("scale", scale)
nc.write_2d_field("misfit", misfit)

nc.close()



