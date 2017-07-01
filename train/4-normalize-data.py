#!/usr/bin/python

import os
import sys
import cv2
import math
import osgeo
import subprocess
import numpy as np

from osgeo import osr
from osgeo import ogr
from osgeo import gdal
from os.path import basename


def meansub_n( raster, mean = 0 ):

  if ( mean == 0 ):
    nsum = np.nansum( raster, axis = ( 0,1 ) )
    mean = nsum / ( ( raster.shape[0] * raster.shape[1] ) - np.count_nonzero(np.isnan(raster)) )

  raster -= mean

  return raster.astype(np.float32)

def stretch_n( raster, lower_percent = 2.5, higher_percent = 97.5, c = 0, d = 0 ):

  a = 0.0
  b = 1.0

  if ( (c == 0) and (d == 0) ):
    c = np.nanpercentile(raster[:, :], lower_percent)
    d = np.nanpercentile(raster[:, :], higher_percent)

  raster = a + (raster[:, :] - c) * (b - a) / (d - c)
  raster[raster < a] = a
  raster[raster > b] = b

  return raster.astype(np.float32)


# raster files
files = ["work/S1*SW?.TIF",
         "work/S2*_??????.TIF"]

for f in files:

  # list tiles
  tiles = subprocess.check_output( "ls %s" % f, shell = True )

  # iterate tiles
  for tile in tiles.split("\n"):
    if (len(tile) > 0):

      print tile

      I = gdal.Open( tile )

      geo = I.GetGeoTransform()
      srs = osr.SpatialReference()
      srs.ImportFromWkt( I.GetProjectionRef() )

      if ( os.path.isfile( tile.split('.')[0] + "-nrm.TIF" ) ):
        continue

      # create new raster file
      drv = gdal.GetDriverByName('GTiff')
      out = drv.Create( tile.split('.')[0] + "-nrm.TIF", I.RasterXSize, I.RasterYSize, I.RasterCount, gdal.GDT_Float32, [ 'TILED=YES' ] )

      # copy geo property
      out.SetGeoTransform( geo )
      out.SetProjection( srs.ExportToWkt() )

      for ch in range( I.RasterCount ):

        print "  normalize band: #", ch+1

        band = np.array( I.GetRasterBand(ch+1).ReadAsArray().astype( np.float32 ) )

        band[ band == 0.0 ] = np.nan

        # [work/s1a.vrt] band #1 c=0.01584162190556526184082031250000 d=0.47834372296929217327488004229963 mean=0.21321091999601360367222468994441
        # [work/s1a.vrt] band #2 c=0.00430294359102845191955566406250 d=0.11530505511909705518291957559995 mean=0.23031137343057539457191751353093
        if ("S1" in f):
          print "apply s1 norm"
          c=[ 0.01584162190556526184082031250000, 0.00430294359102845191955566406250 ]
          d=[ 0.47834372296929217327488004229963, 0.11530505511909705518291957559995 ]
          m=[ 0.21321091999601360367222468994441, 0.23031137343057539457191751353093 ]

          #[work/s2a.vrt] band #1 c=82.00000000000000000000000000000000 d=2765.00000000000000000000000000000000 mean=0.33236541367744976138709489532630
          #[work/s2a.vrt] band #2 c=252.00000000000000000000000000000000 d=1835.00000000000000000000000000000000 mean=0.38469272989922170813414936674235
          #[work/s2a.vrt] band #3 c=84.00000000000000000000000000000000 d=1285.00000000000000000000000000000000 mean=0.36745743511117534563226172394934
          #[work/s2a.vrt] band #4 c=720.00000000000000000000000000000000 d=5339.00000000000000000000000000000000 mean=0.56603211993150448488876236297074

        if ("S2" in f):
          print "apply s2 norm"
          c=[   82.0,  252.0,   84.0,  720.0 ]
          d=[ 2765.0, 1835.0, 1285.0, 5339.0 ]
          m=[ 0.33236541367744976138709489532630, 0.38469272989922170813414936674235, 0.36745743511117534563226172394934, 0.56603211993150448488876236297074 ]

        band = meansub_n( stretch_n( band, 2.5, 97.5, c[ch], d[ch] ), m[ch] )
        band[ np.isnan( band ) ] = 0.0

        print "  write out band: #", ch+1

        outband = out.GetRasterBand( ch+1 )
        outband.WriteArray( band )

        # set nodata
        outband.SetNoDataValue(0.0)

        outband.FlushCache()

