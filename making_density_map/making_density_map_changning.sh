#bash making_density_map_changning.sh ../catalog/lola_date_hour_filter_changning.txt ../density_map_changning

input=$1
output=$2


if [ -d $output ];then
    echo "$output exist"
else
    mkdir $output
fi

python making_density_map.py --input $input --output $output --minlo 104.2 --maxlo 105.4 --minla 27.8 --maxla 28.6 

bash batch_change_projection_changning.sh $output