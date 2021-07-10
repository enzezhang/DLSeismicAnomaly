#!/bin/bash
network=$1

para_file=para.ini
para_py=./script/parameters.py
work_root=$(python2 ${para_py} -p ${para_file} working_root)



python ${work_root}/inference_5days.py $work_dir $work_dir/para.ini ${work_root}/list/inference_changning_daily_5days.txt --workers 1 --batchSize 1 --cuda  --resume $work_dir/$network


