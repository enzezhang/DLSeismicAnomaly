data_dir=../density_map_changning

if [ -f ./changning_original_images.txt ];then
	echo "iamge list exist"
else
	find ${data_dir}/*lola.tif > ./changning_original_images.txt 
fi

if [ -d ${data_dir}/aug ];then
	echo "aug exist"
else
	mkdir ${data_dir}/aug
fi

python image_augment.py ./changning_original_images.txt -o ${data_dir}/aug
