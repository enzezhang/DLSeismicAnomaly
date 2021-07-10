
label=$1
para_file=para.ini
para_py=../parameters.py
data_path=../../density_map_changning

work_root=$(python2 ${para_py} -p ${para_file} working_root)

find ${data_path}/*_lola.tif > image.txt
python ${work_root}/making_pairs_train_5days.py --image image.txt --label $label > ori_pairs.txt


find ${data_path}/aug/*lola_R180.tif > R180.txt
python ${work_root}/making_pairs_train_5days.py --image R180.txt --label $label > R180_pairs.txt

find ${data_path}/aug/*_lola_R270.tif > R270.txt
python ${work_root}/making_pairs_train_5days.py --image R270.txt --label $label > R270_pairs.txt

find ${data_path}/aug/*_lola_R90.tif > R90.txt
python ${work_root}/making_pairs_train_5days.py --image R90.txt --label $label > R90_pairs.txt

find ${data_path}/aug/*_lola_R45.tif > R45.txt
python ${work_root}/making_pairs_train_5days.py --image R45.txt --label $label > R45_pairs.txt

find ${data_path}/aug/*_lola_R135.tif > R135.txt
python ${work_root}/making_pairs_train_5days.py --image R135.txt --label $label > R135_pairs.txt

find ${data_path}/aug/*_lola_R225.tif > R225.txt
python ${work_root}/making_pairs_train_5days.py --image R225.txt --label $label > R225_pairs.txt

find ${data_path}/aug/*_lola_R315.tif > R315.txt
python ${work_root}/making_pairs_train_5days.py --image R315.txt --label $label > R315_pairs.txt

find ${data_path}/aug/*fliplr.tif > fliplr.txt
python ${work_root}/making_pairs_train_5days.py --image fliplr.txt --label $label > fliplr_pairs.txt

find ${data_path}/aug/*Right20.tif > right_20.txt 
python ${work_root}/making_pairs_train_5days.py --image right_20.txt --label $label > right_20_pairs.txt

find ${data_path}/aug/*Left20.tif > left_20.txt 
python ${work_root}/making_pairs_train_5days.py --image left_20.txt --label $label > left_20_pairs.txt

find ${data_path}/aug/*Right10.tif > right_10.txt
python ${work_root}/making_pairs_train_5days.py --image right_10.txt --label $label > right_10_pairs.txt     
find ${data_path}/aug/*Left10.tif > left_10.txt
python ${work_root}/making_pairs_train_5days.py --image left_10.txt --label $label > left_10_pairs.txt


find ${data_path}/aug/*Right15.tif > right_15.txt
python ${work_root}/making_pairs_train_5days.py --image right_15.txt --label $label > right_15_pairs.txt
find ${data_path}/aug/*Left15.tif > left_15.txt
python ${work_root}/making_pairs_train_5days.py --image left_15.txt --label $label > left_15_pairs.txt

find ${data_path}/aug/*Right5.tif > right_5.txt
python ${work_root}/making_pairs_train_5days.py --image right_5.txt --label $label > right_5_pairs.txt

find ${data_path}/aug/*Left5.tif > left_5.txt
python ${work_root}/making_pairs_train_5days.py --image left_5.txt --label $label > left_5_pairs.txt 





find ${data_path}/aug/*Right25.tif > right_25.txt
python ${work_root}/making_pairs_train_5days.py --image right_25.txt --label $label > right_25_pairs.txt    

find ${data_path}/aug/*Left25.tif > left_25.txt
python ${work_root}/making_pairs_train_5days.py --image left_25.txt --label $label > left_25_pairs.txt

find ${data_path}/aug/*Up10.tif >up_10.txt
python ${work_root}/making_pairs_train_5days.py --image up_10.txt --label $label > up_10_pairs.txt
find ${data_path}/aug/*Down10.tif > down_10.txt
python ${work_root}/making_pairs_train_5days.py --image down_10.txt --label $label > down_10_pairs.txt

find ${data_path}/aug/*Up5.tif >up_5.txt
python ${work_root}/making_pairs_train_5days.py --image up_5.txt --label $label > up_5_pairs.txt
find ${data_path}/aug/*Down5.tif > down_5.txt
python ${work_root}/making_pairs_train_5days.py --image down_5.txt --label $label > down_5_pairs.txt

cat ori_pairs.txt R180_pairs.txt R270_pairs.txt R90_pairs.txt R45_pairs.txt R135_pairs.txt R225_pairs.txt R315_pairs.txt right_20_pairs.txt left_20_pairs.txt right_10_pairs.txt left_10_pairs.txt right_5_pairs.txt left_5_pairs.txt left_15_pairs.txt right_15_pairs.txt right_25_pairs.txt left_25_pairs.txt up_10_pairs.txt down_10_pairs.txt up_5_pairs.txt down_5_pairs.txt > all_pairs_1-3_droprate_Feb23_add_mv_aug_H5-20_V5-10.txt
