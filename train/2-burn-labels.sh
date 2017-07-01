#!/bin/bash

rm -rf data/label/*

for s in `ls data/vector/*.sqlite`
do

    name=$(echo $s | xargs -n1 basename)
    tile=$(echo ${name/.sqlite} | cut -d'-' -f2)

    echo $tile

    gdal_rasterize -a_nodata 255 -ot Byte -co "TILED=YES" \
                   -tr 10 10 \
                   -init 255 -a classid \
                   ${s} data/label/${tile}.TIF

done
