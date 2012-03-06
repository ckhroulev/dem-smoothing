#!/usr/bin/env python
import numpy as np
import leastsq
from optparse import OptionParser

try:
    from netCDF4 import Dataset as NC
except:
    from netCDF3 import Dataset as NC

## Set up the option parser
parser = OptionParser()
parser.usage = "usage: %prog n_levels input_file output_file"
parser.description = """\
This script computes smoothed surface elevation maps and saves data in
output_file."""

(options, args) = parser.parse_args()

if len(args) == 3:
    n_levels = int(args[0])
    input_file = args[1]
    output_file = args[2]
else:
    print('wrong number of arguments, 2 expected')
    parser.print_help()
    exit(0)

nc = NC(input_file)

x = nc.variables['x'][:]
y = nc.variables['y'][:]
dem = np.squeeze(nc.variables['usurf'][:])

nc.close()

# x and y better be strictly increasing
dx = x[1] - x[0]
dy = y[1] - y[0]

nc = NC(output_file, 'w')

nc.createDimension('x', x.size)
nc.createDimension('y', y.size)
nc.createDimension('level', n_levels)
nc.createDimension('coeff', leastsq.n_params)

data = nc.createVariable('data', 'f8', ('y', 'x', 'level', 'coeff'))

data[:] = leastsq.smooth(dem, dx, dy, n_levels)

nc.close()
