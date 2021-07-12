# DLSeismicAnomaly
## Installation
```shell
git clone https://github.com/enzezhang/DLSeismicAnomaly.git
cd DLSeismicAnomaly
```
## Install anaconda2
https://repo.anaconda.com/archive/

Choose Anaconda2-2019.10-Linux-x86_64.sh
```Shell
bash Anaconda2-2019.10-Linux-x86_64.sh
```

## Install CUDA
## Install GDAL

```Shell
conda install gdal
```
There could be some library issues.
## Install dependencies
pip install tensorboardX tqdm torch torchvision torchsummary pyproj scipy imgaug rasterio

## Step 0: change dir

change the working_root data_weiyuan, and data_changning in file network_training/para.ini

## Step 1: making earthquake distribution maps.

```Shell
cd making_density_map
bash making_density_map_weiyuan.sh ../catalog/date_lola_hour_mag_weiyuan_all.txt ../density_map_weiyuan
bash making_density_map_changning.sh ../catalog/lola_date_hour_filter_changning.txt ../density_map_changning
```

## Step 2: data augmentation
```Shell
bash image_augment.sh
```

## Step 3: Network training

### making the list of training pairs

```Shell
cd ../network_training/list
bash making_training_pairs_5days.sh label_20210524.txt
```
### train the network 
```Shell
bash exe.sh 
```
after running the code, the network file named "resnet34_SGD_0.003_seed1_weight_decay_0.005_1-3_droprate_May24.tar" will be generated.
## Step 4: Infer the abnormal timings

```Shell
cd list
bash making_inference_pairs_5days.sh
cd ../
bash inference_weiyuan.sh resnet34_SGD_0.003_seed1_weight_decay_0.005_1-3_droprate_May24.tar
bash inference_changning.sh resnet34_SGD_0.003_seed1_weight_decay_0.005_1-3_droprate_May24.tar
```
This step will generate abnormal timings for both Weiyuan and Changning.
## Step 5: extracting abnormal locations.
```Shell
cd ../post_process
bash post_processing_changning.sh abnormal_timings_for_changning
bash post_processing_weiyuan_six_hour.sh abnormal_timings_for_weiyuan
```
abnormal_timings_for_changning and abnormal_timings_for_weiyuan need to be specified.



