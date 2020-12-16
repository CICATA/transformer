import rasterio
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import shape
import gdal
import osr
import csv

location = 'L15-0331E-1257N_1327_3160_13'
date = '2018_03'

path = '/data/gee/SAR/data/SN7/train/' + location + '/images/global_monthly_' + date + '_mosaic_' + location + '.tif'

# Read theimage as geo
geoimage = rasterio.open(path)

# Convert coordinates to pixel positions
lat = -121.67141943033411
lon = 37.961579543270666

print(geoimage.index(lat, lon))

# Convert the geo raster to numpy and plot
r = geoimage.read(1)
g = geoimage.read(2)
b = geoimage.read(3)
image = np.dstack((r, g, b))
plt.imshow(image)
plt.show()

# For debug purpouses
print('eof')
