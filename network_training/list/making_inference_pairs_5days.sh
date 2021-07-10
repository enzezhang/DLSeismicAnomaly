
label=$1
para_file=para.ini
para_py=../parameters.py
data_path=../../density_map_weiyuan

work_root=$(python2 ${para_py} -p ${para_file} working_root)

find ${data_path}/*_lola.tif > image.txt
python ${work_root}/making_pairs_inference_5days.py --image image.txt > inference_weiyuan_6_hour_5days.txt

data_path=../../density_map_changning

find ${data_path}/*_lola.tif > image.txt
python ${work_root}/making_pairs_inference_5days.py --image image.txt > inference_changning_daily_5days.txt
