
output=$1
#python /home/zez/data2/DL_identify_earthquake_distribution_change/calculate_F1.py --output $output --label /home/zez/data2/identify_earthquake_distribution_change/weiyuan/list/label_weiyuan_Feb22.txt
#python /home/zez/data2/DL_identify_earthquake_distribution_change/calculate_F1.py --output $output --label /home/zez/data2/identify_earthquake_distribution_change/weiyuan/new_data/all_data/list/label_20200321.txt

#cat $output| head -n 365 | tail -n +245 > 201905.txt 
#echo "------201905-----------"
#python /home/zez/data2/DL_identify_earthquake_distribution_change/calculate_F1.py --output 201905.txt --label /home/zez/data2/identify_earthquake_distribution_change/weiyuan/new_data/jinping/label/201905_label_Apr30.txt

#echo "------201909-----------"
#cat $output| head -n 829 | tail -n +714 > 201909.txt

#python /home/zez/data2/DL_identify_earthquake_distribution_change/calculate_F1.py --output 201909.txt --label /home/zez/data2/identify_earthquake_distribution_change/weiyuan/new_data/jinping/label/201909_label_Apr30.txt

echo "-------201903_201910--------"
cat $output |head -n 937 > 201903_10.txt
#cat 201905.txt 201909.txt > 201905_20190905.txt
python /home/zez/data2/DL_identify_earthquake_distribution_change/calculate_F1.py --output 201903_10.txt --label /home/zez/data2/identify_earthquake_distribution_change/weiyuan/new_data/jinping/label/201903_10_label_May9.txt
