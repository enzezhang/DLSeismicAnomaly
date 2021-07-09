#!/bin/bash

path=$1
#gdalwarp -s_srs EPSG:32648 -t_srs EPSG:4326 out.tif out_lola.tif

cd $path
file=(`ls *.tif`)
count=${#file[@]}
i=0
while (($i<$count))
do
	temp=(`echo ${file[i]}| cut -d '.' -f 1`)
	echo "gdalwarp -s_srs '+proj=utm +zone=48 +datum=WGS84' -t_srs '+proj=utm +zone=48 +datum=WGS84' -ts 38 33 ${file[i]} ${temp}_UTM48.tif"
	gdalwarp -s_srs '+proj=utm +zone=48 +datum=WGS84' -t_srs '+proj=utm +zone=48 +datum=WGS84' -ts 38 33 ${file[i]} ${temp}_UTM48.tif
	echo "gdalwarp -s_srs '+proj=utm +zone=48 +datum=WGS84' -t_srs EPSG:4326 -ts 38 33 ${temp}_UTM48.tif ${temp}_lola.tif"
	gdalwarp -s_srs '+proj=utm +zone=48 +datum=WGS84' -t_srs EPSG:4326 -ts 38 33  ${temp}_UTM48.tif ${temp}_lola.tif
	i=$[i+1]
done
