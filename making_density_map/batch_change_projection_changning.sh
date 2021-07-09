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
	gdalwarp -s_srs '+proj=utm +zone=48 +datum=WGS84' -t_srs '+proj=utm +zone=48 +datum=WGS84' -ts 59 44 ${file[i]} ${temp}_UTM48.tif
	gdalwarp -s_srs '+proj=utm +zone=48 +datum=WGS84' -t_srs EPSG:4326 -ts 59 44 ${temp}_UTM48.tif ${temp}_lola.tif
	#gdal_calc.py -A ${temp}_lola.tif --outfile=${temp}_lola_new.tif --calc="A*(A>0.01)" --NoDataValue=0
	i=$[i+1]
done
