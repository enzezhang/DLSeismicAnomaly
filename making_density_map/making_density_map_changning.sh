#python making_density_map.py --input /home/zez/data2/identify_earthquake_distribution_change/data_MentLY/lola_date_hour_filter.txt --output /home/zez/data2/identify_earthquake_distribution_change/data_MentLY/density_map_new --minlo 104.2 --maxlo 105.4 --minla 27.8 --maxla 28.6


input=$1
output=$2
python making_density_map.py --input $input --output $output --minlo 104.2 --maxlo 105.4 --minla 27.8 --maxla 28.6 

#python making_density_map.py --input /home/zez/data2/identify_earthquake_distribution_change/data_MentLY/test_post_processing/final_out_lola_resnet18_SGD_momentum_0_0.001_seed1_weight_decay_0.01_1-3_droprate_add_mv_aug_H5-25_V5-10_Feb23inference_pairs_changning_5day.txt --output /home/zez/data2/identify_earthquake_distribution_change/data_MentLY/post_process_map_resnet18_SGD_momentum_0_0.001_seed1_weight_decay_0.01_1-3_droprate_add_mv_aug_H5-25_V5-10_Feb23inference_pairs_changning_5day --minlo 104.2 --maxlo 105.4 --minla 27.8 --maxla 28.6
#python making_density_map.py --input /home/zez/data2/identify_earthquake_distribution_change/weiyuan/new_data/all_data/date_lola_mag_all.txt --output /home/zez/data2/identify_earthquake_distribution_change/weiyuan/new_data/all_data/density_map --minlo 104.21 --maxlo 105 --minla 29.2 --maxla 29.8
