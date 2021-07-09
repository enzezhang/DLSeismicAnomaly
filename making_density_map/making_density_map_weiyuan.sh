
#bash making_density_map_weiyuan.sh ../catalog/date_lola_hour_mag_weiyuan_all.txt ../density_map_weiyuan
catalog=$1
output=$2

if [ -d $output ];then
    echo "$output exist"
else
    mkdir $output
fi
python making_density_map_hourly.py --input $catalog --output $output --resolution 6 --minlo 104.21 --maxlo 105 --minla 29.2 --maxla 29.8

bash batch_change_projection_weiyuan.sh $output
