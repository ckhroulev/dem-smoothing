#!/usr/bin/env python
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset as NC
from pyproj import Proj

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sys

print "Trying to open '%s'" % sys.argv[1]
nc = NC(sys.argv[1], 'r')

# we need to know longitudes and latitudes corresponding to our grid
proj = Proj(nc.projection)

x = nc.variables['x'][:]
y = nc.variables['y'][:]
xx,yy = np.meshgrid(x,y)

lon,lat = proj(xx,yy,inverse=True)

# x and y *in the dataset* are only used to determine plotting domain
# dimensions
width = x.max() - x.min()
height = y.max() - y.min()

# load data
usurf = np.squeeze(nc.variables['usurf'][:])

m = Basemap(width=width,      # width in projection coordinates, in meters
            height=height,      # height
            resolution='h',     # coastline resolution, can be 'l' (low), 'h'
                                # (high) and 'f' (full)
            projection='stere', # stereographic projection
            lat_ts=71,          # latitude of true scale
            lon_0=-41,          # longitude of the plotting domain center
            lat_0=72)           # latitude of the plotting domain center

m.drawcoastlines(color='red')

# convert longitudes and latitudes to x and y:
xx,yy = m(lon, lat)

m.contour(xx, yy, usurf, 41)

# draw parallels and meridians. The labels argument specifies where to draw
# ticks: [left, right, top, bottom]
m.drawparallels(np.arange(-55.,90.,5.), labels = [1, 0, 0, 0])
m.drawmeridians(np.arange(-120.,30.,10.), labels = [0, 0, 0, 1])

plt.show()
#plt.savefig('greenland.png')
