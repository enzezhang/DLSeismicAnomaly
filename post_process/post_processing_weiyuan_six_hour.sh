#!/bin/bash

output=$1


image_folder=../density_map_weiyuan

find ${image_folder}/*lola.tif > image_list_weiyuan.txt

final_out_lola=final_output_${output%.txt*}.txt
rm $final_out_lola
date=(`python find_positive_date.py --output $output --list image_list_weiyuan.txt`)
i=0
count=${#date[@]}
echo $count
while (($i<$count))
do
	#echo ${date[i]}
	ori_image=(`find $image_folder/${date[i]}*lola.tif`)
	python extract_location_hour.py --filename $ori_image --threshold 5 --outputfile $final_out_lola --resolution 6
	i=$[i+1]
done	
