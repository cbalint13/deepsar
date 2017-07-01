#!/usr/bin/python

import os
import sys
import cv2
import math
import h5py
import osgeo
import random
import struct
import numpy as np

import matplotlib.pyplot as plt

from osgeo import osr
from osgeo import ogr
from osgeo import gdal

from shapely import geometry
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import cascaded_union, polygonize

# transforms
wgsSR = osr.SpatialReference()
wgsSR.ImportFromEPSG(4326) # WGS84
s70SR = osr.SpatialReference()
s70SR.ImportFromEPSG(31700) # Stereo70
wgsTOs70 = osr.CoordinateTransformation(wgsSR,s70SR)
s70TOwgs = osr.CoordinateTransformation(s70SR,wgsSR)


psize = 224 # patch size
patches = 120000
h5samples = 1000000 # samples

nLastTick=-1
def TermProgress( dfComplete, pszMessage, pProgressArg ):

  global nLastTick
  nThisTick = (int) (dfComplete * 40.0)

  if nThisTick < 0:
    nThisTick = 0
  if nThisTick > 40:
    nThisTick = 40

  # Have we started a new progress run?
  if nThisTick < nLastTick and nLastTick >= 39:
    nLastTick = -1

  if nThisTick <= nLastTick:
    return True

  while nThisTick > nLastTick:
    nLastTick = nLastTick + 1
    if (nLastTick % 4) == 0:
      sys.stdout.write('%d' % ((nLastTick / 4) * 10))
    else:
      sys.stdout.write('.')

  if nThisTick == 40:
    print(" - done." )
  else:
    sys.stdout.flush()

  return True

def display( s1ch, lbch ):

      ax = plt.subplot(211)
      bx = plt.subplot(212)

      rgb = np.zeros((s1ch[0,:,:].shape[0], s1ch[0,:,:].shape[1], 3))
      rgb[:,:,0] = s1ch[0,:,:] - np.min(s1ch[0,:,:])
      rgb[:,:,0] = rgb[:,:,0] * (255.0 / np.max(rgb[:,:,0]))
      rgb[:,:,1] = s1ch[1,:,:] - np.min(s1ch[1,:,:])
      rgb[:,:,1] = rgb[:,:,1] * (255.0 / np.max(rgb[:,:,1]))
      rgb[:,:,2] = (rgb[:,:,0] + rgb[:,:,1]) / 2.0
      rgb = rgb.astype(np.uint8)

      ax.imshow(rgb, origin='upper')

      print lbch.shape
      bx.imshow(lbch[0,:,:], origin='upper')

      plt.draw()
      plt.waitforbuttonpress(0)
      #plt.close()

def getpatch( iDst, x, y, angle, scale ):

  channs = iDst.RasterCount

  patch = np.zeros( ( channs, psize, psize ), dtype=np.float32 )

  # radians
  angle *= np.pi / 180.0
  # transforms
  tsin = np.sin(angle) * scale;
  tcos = np.cos(angle) * scale;

  hcols = psize / 2.0
  hrows = psize / 2.0

  xoff, yoff = -hcols, -hrows
  p1x = x + xoff*tcos - yoff*tsin
  p1y = y + xoff*tsin + yoff*tcos
  xoff, yoff = +hcols, -hrows
  p2x = x + xoff*tcos - yoff*tsin
  p2y = y + xoff*tsin + yoff*tcos
  xoff, yoff = +hcols, +hrows
  p3x = x + xoff*tcos - yoff*tsin
  p3y = y + xoff*tsin + yoff*tcos
  xoff, yoff = -hcols, +hrows
  p4x = x + xoff*tcos - yoff*tsin
  p4y = y + xoff*tsin + yoff*tcos

  bminx = min( [p1x, p2x, p3x, p4x] )
  bmaxx = max( [p1x, p2x, p3x, p4x] )
  bminy = min( [p1y, p2y, p3y, p4y] )
  bmaxy = max( [p1y, p2y, p3y, p4y] )

  for ch in range(0, channs):

    band = iDst.GetRasterBand(ch+1)
    valueType = {osgeo.gdal.GDT_Byte: 'B', osgeo.gdal.GDT_UInt16: 'H', osgeo.gdal.GDT_Float32: 'f'}[band.DataType]

    hsize = int(bmaxx-bminx)+2
    vsize = int(bmaxy-bminy)+2

    # force square
    rsize=max(hsize,vsize)

    vsize = rsize
    hsize = rsize

    values = struct.unpack( '%d%s' % ( hsize*vsize, valueType ), band.ReadRaster( int(bminx), int(bminy), hsize, vsize ) )
    sample = np.reshape( values, (hsize, vsize) )

    for px in range( 0, psize):
      for py in range( 0, psize):

        xoff = px - hcols
        yoff = py - hrows

        # the rotation shifts & scale
        img_x = int( (vsize / 2.0) + xoff*tcos - yoff*tsin )
        img_y = int( (hsize / 2.0) + xoff*tsin + yoff*tcos )

        try:
          patch[ ch, py, px ] = sample[img_y, img_x]
        except:
          print "Error: ", patch.shape, sample.shape, img_y, img_x
          sys.exit(-1)

  return patch

def getpatch2( iDst, x, y ):

  channs = iDst.RasterCount

  patch = np.zeros( ( channs, psize, psize ), dtype=np.float32 )

  for ch in range(0, channs):

    band = iDst.GetRasterBand(ch+1)
    valueType = {osgeo.gdal.GDT_Byte: 'B', osgeo.gdal.GDT_UInt16: 'H', osgeo.gdal.GDT_Float32: 'f'}[band.DataType]

    values = struct.unpack( '%d%s' % ( psize*psize, valueType ), band.ReadRaster( int(x-psize/2), int(y-psize/2), psize, psize ) )
    patch[ch, :, :] = np.reshape( values, (psize, psize) )

  return patch

# N x C x H x W data
# N x C x H x W label


# radar
s1_src = "work/limits/S1A_IW_SLC__1SDV.shp"
s1_rst = gdal.Open("work/sentinel1-train.vrt")

# label
lb_src = "data/label/label.shp"
lb_rst = gdal.Open("data/label/label.vrt")


s1geotrans = s1_rst.GetGeoTransform()
lbgeotrans = lb_rst.GetGeoTransform()

s1shp = ogr.Open(s1_src, 0)
lbshp = ogr.Open(lb_src, 0)

s1lyr = s1shp.GetLayer()
lblyr = lbshp.GetLayer()

s1exts = s1lyr.GetExtent()
s1pl = s1lyr.GetNextFeature()
s1geom = s1pl.GetGeometryRef()
s1minx = min( s1exts[0], s1exts[1] )
s1maxx = max( s1exts[0], s1exts[1] )
s1miny = min( s1exts[2], s1exts[3] )
s1maxy = max( s1exts[2], s1exts[3] )

lbexts = lblyr.GetExtent()
lbpl = lblyr.GetNextFeature()
lbgeom = lbpl.GetGeometryRef()
lbminx = min( s1exts[0], s1exts[1] )
lbmaxx = max( s1exts[0], s1exts[1] )
lbminy = min( s1exts[2], s1exts[3] )
lbmaxy = max( s1exts[2], s1exts[3] )


plt.figure(figsize = (1,3))

cnt = 0
h5_idx = 0
in_idx = 0
ptxy = np.zeros( (3), dtype='f' )
while (cnt < patches):

  gx = random.uniform(s1minx, s1maxx)
  gy = random.uniform(s1miny, s1maxy)

  for a in range(0,360,360):

    # prepare sample rectangle
    angle = a * np.pi / 180.0
    tsin = np.sin(angle) * 1.0;
    tcos = np.cos(angle) * 1.0;

    hx = psize*10.0 / 2.0
    hy = psize*10.0 / 2.0

    xoff, yoff = -hx, -hy
    p1x = gx + xoff*tcos - yoff*tsin
    p1y = gy + xoff*tsin + yoff*tcos

    xoff, yoff = +hx, -hy
    p2x = gx + xoff*tcos - yoff*tsin
    p2y = gy + xoff*tsin + yoff*tcos

    xoff, yoff = +hx, +hy
    p3x = gx + xoff*tcos - yoff*tsin
    p3y = gy + xoff*tsin + yoff*tcos

    xoff, yoff = -hx, +hy
    p4x = gx + xoff*tcos - yoff*tsin
    p4y = gy + xoff*tsin + yoff*tcos

    patch = ogr.CreateGeometryFromWkt( Polygon([[p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y]]).to_wkt() )

    if ( ( patch.Within( s1geom ) ) and
         ( patch.Within( lbgeom ) ) ):

      # compute pixel x,y from S1 geocoord
      px = (gx - s1geotrans[0]) / s1geotrans[1]
      py = (gy - s1geotrans[3]) / s1geotrans[5]
      # sample
      #s1ch = getpatch( s1_rst, px, py, a, 1.0 )
      s1ch = getpatch2( s1_rst, px, py )
      if ( s1ch.size == 0 ):
        #print "skip s1ch ",x,y
        continue

      # compute pixel x,y from lb geocoord
      px = (gx - lbgeotrans[0]) / lbgeotrans[1]
      py = (gy - lbgeotrans[3]) / lbgeotrans[5]
      # sample

      #lbch = getpatch( lb_rst, px, py, a, 1.0 )
      lbch = getpatch2( lb_rst, px, py )
      if ( lbch.size == 0 ):
        #print "skip lbch ",x,y
        continue

      lbch[lbch==255] = 0

      lbch[lbch== 7] = 255
      lbch[lbch== 9] = 255
      lbch[lbch==17] = 255
      lbch[lbch==28] = 255

      lbch[lbch!=255] = 0
      lbch[lbch==255] = 1

      if ( np.sum(lbch) < 512 ):
        continue

      # debug
#      display(s1ch, lbch)


      ptxy[0] = gx
      ptxy[1] = gy
      ptxy[2] = a

      # new hdf5 file
      if ((in_idx >= h5samples) or (h5_idx == 0)):
        in_idx = 0
        h5_id = '{:08}'.format(h5_idx)
        h5f = h5py.File('./hdf5-new/dataset-%s-lb.h5' % h5_id)
        rset = h5f.create_dataset('radar', (h5samples, s1ch.shape[0], s1ch.shape[1], s1ch.shape[2]), chunks=(1, s1ch.shape[0], s1ch.shape[1], s1ch.shape[2]), maxshape=(None, s1ch.shape[0], s1ch.shape[1], s1ch.shape[2]), dtype="f")
        iset = h5f.create_dataset('label', (h5samples, lbch.shape[0], lbch.shape[1], lbch.shape[2]), chunks=(1, lbch.shape[0], lbch.shape[1], lbch.shape[2]), maxshape=(None, lbch.shape[0], lbch.shape[1], lbch.shape[2]), dtype="f")
        pset = h5f.create_dataset('coord', (h5samples, ptxy.shape[0]), chunks=(1, ptxy.shape[0]), maxshape=(None, ptxy.shape[0]), dtype="f")
        h5_idx += 1

      rset[ in_idx, ... ] = s1ch
      iset[ in_idx, ... ] = lbch
      pset[ in_idx, ... ] = ptxy

      cnt += 1
      in_idx += 1

      TermProgress( float( cnt ) / float( patches ), None, None )

TermProgress( 1.0, None, None )

rset.resize((in_idx, s1ch.shape[0], psize, psize))
iset.resize((in_idx, lbch.shape[0], psize, psize))
pset.resize((in_idx, ptxy.shape[0]))

