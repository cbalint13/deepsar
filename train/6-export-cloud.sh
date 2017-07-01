#!/bin/bash


rm -f work/clouds/*.sqlite

for c in `find ./data/s2a/S2A_USER_PRD_MSIL2A*/ -type f -name "*CLOUD*.gml"`;
do

   count=`ogrinfo $c -al | grep Count | awk {'print $3'}`

   if [ ! "$count" = "" ];
   then


     dn=$(echo $c | xargs -n1 dirname)
     fn=$(echo $c | xargs -n1 basename)

     echo [$fn] [$count]

     pushd $dn
       srs=$(gdalsrsinfo -o proj4 *CLD*.jp2 | sed -e "s|[']||g")
       scene=`ls ../../../*.xml | grep MTD | xargs -n1 basename`
     popd

     ogr2ogr -update -append -f 'SQLite' \
             -s_srs "$srs" -t_srs EPSG:31700 \
             -a_srs EPSG:31700 -nln clouds \
             work/clouds/${scene/.xml}.sqlite $c

   fi

done


