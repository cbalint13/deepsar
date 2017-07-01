#!/bin/bash

rm -rf data/vector/*.sqlite*

# split into classes
for c in `cat data/landcover/classes.txt | sed -e 's| |_|g'`;
do

  clsid=$(echo $c | cut -d":" -f1)
  clslb=$(echo $c | cut -d":" -f2 | sed -e 's|_| |g')

  echo "CLASS: [$clsid] [$clslb]"

  # feed all shapefiles
  for s in `ls data/landcover/*.shp| grep -v 'footprint'`;
  do

    table=$(echo $s | xargs -n1 basename)

    # split into grid tiles
    for tile in `ogrinfo -al data/grid/epsg3395_50000.shp | grep LABEL | grep _ | awk '{print $4}'`
    do

      echo " [$s] [$table] [$tile]"

      ogr2ogr -update -append -f 'SQLite' \
              -t_srs EPSG:3395 -nln land \
              -dialect sqlite \
              -clipdst data/grid/epsg3395_50000.shp \
              -clipdstwhere "LABEL='${tile}'" \
              -sql "SELECT CAST('$clsid' AS int) AS classid, geometry FROM '${table/.shp}' WHERE LC_CLASS_E='$clslb'" \
              data/vector/${tile}.sqlite $s

    done

  done

done

