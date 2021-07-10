#!/bin/bash


para_file=para.ini
para_py=./script/parameters.py
work_root=$(python2 ${para_py} -p ${para_file} working_root)

lr=0.003
wd=0.005
network=resnet34_SGD_${lr}_seed1_weight_decay_${wd}_1-3_droprate_May24.tar
rm $network.txt

python ${work_root}/train_5day.py ${work_root} ${work_root}/para.ini ${work_root}/list/all_pairs_1_3_droprate_May24.txt --lr $lr --momentum 0.0 --weight-decay $wd --workers 1 --epochs 1000 --seed 1 --patience 10 --batch-size 32 --checkname $network --ft --eval-interval 1
