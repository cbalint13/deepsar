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

def display( s1ch, s2ch ):

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

      rgb = np.zeros((s2ch[0,:,:].shape[0], s2ch[0,:,:].shape[1], 3))
      rgb[:,:,0] = s2ch[0,:,:] - np.min(s2ch[0,:,:])
      rgb[:,:,0] = rgb[:,:,0] * (255.0 / np.max(rgb[:,:,0]))
      rgb[:,:,1] = s2ch[1,:,:] - np.min(s2ch[1,:,:])
      rgb[:,:,1] = rgb[:,:,1] * (255.0 / np.max(rgb[:,:,1]))
      rgb[:,:,2] = s2ch[2,:,:] - np.min(s2ch[2,:,:])
      rgb[:,:,2] = rgb[:,:,2] * (255.0 / np.max(rgb[:,:,2]))
      rgb = rgb.astype(np.uint8)

      bx.imshow(rgb, origin='upper')

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

# image
s2_src = "work/limits/S2A_OPER_MTD_L1C_TL_SGS__20160713T125925_A005524.shp"
s2_cld = "work/clouds/S2A_USER_MTD_SAFL2A_PDMC_20160713T144304_R093_V20160713T092032_20160713T092032.shp"
s2_rst = gdal.Open("work/sentinel2-train.vrt")


s1geotrans = s1_rst.GetGeoTransform()
s2geotrans = s2_rst.GetGeoTransform()

s1shp = ogr.Open(s1_src, 0)
s2shp = ogr.Open(s2_src, 0)
c2shp = ogr.Open(s2_cld, 0)

s1lyr = s1shp.GetLayer()
s2lyr = s2shp.GetLayer()
c2lyr = c2shp.GetLayer()

s1exts = s1lyr.GetExtent()
s1pl = s1lyr.GetNextFeature()
s1geom = s1pl.GetGeometryRef()
s1minx = min( s1exts[0], s1exts[1] )
s1maxx = max( s1exts[0], s1exts[1] )
s1miny = min( s1exts[2], s1exts[3] )
s1maxy = max( s1exts[2], s1exts[3] )

s2exts = s2lyr.GetExtent()
s2pl = s2lyr.GetNextFeature()
s2geom = s2pl.GetGeometryRef()
s2minx = min( s1exts[0], s1exts[1] )
s2maxx = max( s1exts[0], s1exts[1] )
s2miny = min( s1exts[2], s1exts[3] )
s2maxy = max( s1exts[2], s1exts[3] )

s2cl = c2lyr.GetNextFeature()
c2geom = s2cl.GetGeometryRef()


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
         ( patch.Within( s2geom ) ) ):

      if ( patch.Intersect( c2geom ) ):
        continue

      # compute pixel x,y from S1 geocoord
      px = (gx - s1geotrans[0]) / s1geotrans[1]
      py = (gy - s1geotrans[3]) / s1geotrans[5]
      # sample
      #s1ch = getpatch( s1_rst, px, py, a, 1.0 )
      s1ch = getpatch2( s1_rst, px, py )
      if ( s1ch.size == 0 ):
        #print "skip s1ch ",x,y
        continue

      # compute pixel x,y from S2 geocoord
      px = (gx - s2geotrans[0]) / s2geotrans[1]
      py = (gy - s2geotrans[3]) / s2geotrans[5]
      # sample

      #s2ch = getpatch( s2_rst, px, py, a, 1.0 )
      s2ch = getpatch2( s2_rst, px, py )
      if ( s2ch.size == 0 ):
        #print "skip s2ch ",x,y
        continue

      # debug
      #display(s1ch, s2ch)

      ptxy[0] = gx
      ptxy[1] = gy
      ptxy[2] = a

      # new hdf5 file
      if ((in_idx >= h5samples) or (h5_idx == 0)):
        in_idx = 0
        h5_id = '{:08}'.format(h5_idx)
        h5f = h5py.File('./hdf5-new/dataset-%s.h5' % h5_id)
        rset = h5f.create_dataset('radar', (h5samples, s1ch.shape[0], s1ch.shape[1], s1ch.shape[2]), chunks=(1, s1ch.shape[0], s1ch.shape[1], s1ch.shape[2]), maxshape=(None, s1ch.shape[0], s1ch.shape[1], s1ch.shape[2]), dtype="f")
        iset = h5f.create_dataset('image', (h5samples, s2ch.shape[0], s2ch.shape[1], s2ch.shape[2]), chunks=(1, s2ch.shape[0], s2ch.shape[1], s2ch.shape[2]), maxshape=(None, s2ch.shape[0], s2ch.shape[1], s2ch.shape[2]), dtype="f")
        pset = h5f.create_dataset('coord', (h5samples, ptxy.shape[0]), chunks=(1, ptxy.shape[0]), maxshape=(None, ptxy.shape[0]), dtype="f")
        h5_idx += 1

      rset[ in_idx, ... ] = s1ch
      iset[ in_idx, ... ] = s2ch
      pset[ in_idx, ... ] = ptxy

      cnt += 1
      in_idx += 1

      TermProgress( float( cnt ) / float( patches ), None, None )

TermProgress( 1.0, None, None )

rset.resize((in_idx, s1ch.shape[0], psize, psize))
iset.resize((in_idx, s2ch.shape[0], psize, psize))
pset.resize((in_idx, ptxy.shape[0]))

