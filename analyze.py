#!/usr/bin/env python
import numpy as np
from optparse import OptionParser

try:
    from netCDF4 import Dataset as NC
except:
    from netCDF3 import Dataset as NC

## Set up the option parser
parser = OptionParser()
parser.usage = "usage: %prog misfit_file output_file"
parser.description = """\
This script computes the spatial scale minimizing the misfit."""

(options, args) = parser.parse_args()

if len(args) == 2:
    data_file = args[0]
    output_file = args[1]
else:
    print('wrong number of arguments, 2 expected')
    parser.print_help()
    exit(0)

nc_data = NC(data_file, 'r')

# Get data
# note the lack of [:]; this might be slower, but avoids keeping several Gb of
# data in memory
misfit = nc_data.variables['misfit']

x_size = misfit.shape[1]
y_size = misfit.shape[0]
n_levels = misfit.shape[2]

scale = np.zeros((y_size, x_size))

levels = np.arange(n_levels)
for j in xrange(y_size):
    for i in xrange(x_size):

        # skip areas with very little flow
        if np.any(misfit[j,i] < 0):
            scale[j,i] = -1
            continue

        min_misfit = 180.0
        min_misfit_scale = -1
        for k in xrange(n_levels):
            if misfit[j,i,k] < min_misfit:
                min_misfit = misfit[j,i,k]
                min_misfit_scale = n_levels - k

        scale[j,i] = min_misfit_scale

import PISMNC

nc = PISMNC.PISMDataset(output_file, 'w')

nc.createDimension("x", x_size)
nc.createDimension("y", y_size)
nc.createDimension("level", n_levels)
data = nc.createVariable('scale', 'f8', ('y', 'x'))

data[:] = scale

nc.close()



