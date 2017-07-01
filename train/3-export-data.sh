#!/bin/bash


for s in `find ./data/sentinel1/4-tci -type d -name "S1*"`;
do

  if [ ! -f $s/*.vrt ];
  then

    pushd $s

      base=$(pwd | xargs -n1 basename)

      gdalbuildvrt -overwrite -separate -srcnodata "0 0 0 0" -vrtnodata "0 0 0 0" \
                   ${base/.data}.vrt *VV*.img *VH*.img
    popd

  fi

done


for d in `find data/sentinel1/ -name "*.vrt" -type f`;
do

  base=$(basename $d)

  if [ -f work/limits/${base/.vrt}.shp ];
  then

    if [ ! -f work/${base/.vrt}.TIF ];
    then

      echo ${base/.vrt}.TIF
      gdalwarp -srcnodata "0 0" -dstnodata "0 0" \
               -wm 4096 -multi \
               -cutline work/limits/${base/.vrt}.shp \
               $d work/${base/.vrt}.TIF -co "TILED=YES"
    fi

  fi

done



for s in `find ./data/sentinel2/S2*L2A*/GRANULE -type d -name "R10m"`;
do

  base=`ls $s/*B02*.jp2 | sed -e 's|_B02||g' | xargs -n1 basename`

  if [ ! -f $s/${base/_10m.jp2}.vrt ];
  then

    pushd $s
    gdalbuildvrt -overwrite -srcnodata "0 0 0 0" -vrtnodata "0 0 0 0" -separate \
                 ${base/_10m.jp2}.vrt \
                 ${base/_10m.jp2}_B04_10m.jp2 \
                 ${base/_10m.jp2}_B03_10m.jp2 \
                 ${base/_10m.jp2}_B02_10m.jp2 \
                 ${base/_10m.jp2}_B08_10m.jp2
    popd

  fi

done


for s in `find ./data/sentinel2/S2*L2A*/GRANULE -type f -name "*.vrt"`;
do

  base=$(echo $s | xargs -n1 basename)

  if [ ! -f work/${base/.vrt}.TIF ];
  then

    echo ${base/.vrt}
    gdalwarp -srcnodata "0 0 0 0" -dstnodata "0 0 0 0" -tr 10 10 \
             -r bilinear -multi -order 3 -t_srs EPSG:3395 \
             $s work/${base/.vrt}.TIF -co "TILED=YES"

  fi

done


