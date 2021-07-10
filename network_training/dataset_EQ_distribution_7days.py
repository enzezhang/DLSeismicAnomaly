import torch.utils.data as data
import torch
import random
from scipy.ndimage import imread
import os
import os.path
import glob
from PIL import Image
import numpy as np
from skimage import io
HOME = os.path.expanduser('~')
import sys

basicCodes_path = HOME + '/codes/PycharmProjects/DeeplabforRS'
#sys.path.insert(0, basicCodes_path)
import basic_src.basic as  basic
# import split_signal



class patchclass(object):
    """
    store the information of each patch (a small subset of the remote sensing images)
    """
    def __init__(self,org_img,boundary):
        self.org_img = org_img  # the original remote sensing images of this patch
        self.boundary=boundary      # the boundary of patch (xoff,yoff ,xsize, ysize) in pixel coordinate
    def boundary(self):
        return self.boundary

import rasterio


def read_patch(img_path,crop_width):

    #data=np.array(Image.open(img_path))
    data=io.imread(img_path)
    if  data.shape[0] < crop_width or data.shape[1] < crop_width:
        data_expend=np.zeros([crop_width,crop_width])

        data_expend[int((crop_width-data.shape[0])/2):int((crop_width-data.shape[0])/2)+data.shape[0],int((crop_width-data.shape[1])/2):int((crop_width-data.shape[1])/2)+data.shape[1]]=data
        return data_expend
    else:
        return data


def make_dataset(root,list_txt,train=True):
    """
    get the patches information of the remote sensing images. 
    :param root: data root
    :param list_txt: a list file contain images (one row contain one train image and one label 
    image with space in the center if the input is for training; one row contain one image if it is for inference)
    :param patch_w: the width of the expected patch
    :param patch_h: the height of the expected patch
    :param adj_overlay: the extended distance (in pixel) to adjacent patch, make each patch has overlay with adjacent patch
    :param train:  indicate training or inference
    :return: dataset (list)
    """
    dataset = []
    if os.path.isfile(list_txt) is False:
        basic.outputlogMessage("error, file %s not exist"%list_txt)
        assert False

    with open(list_txt) as file_obj:
        files_list = file_obj.readlines()
    if len(files_list) < 1:
        basic.outputlogMessage("error, no file name in the %s" % list_txt)
        assert False
    if train:
        for line in files_list:
            names_list = line.split()
            if len(names_list) < 1: # empty line
                    continue
            sig_name1 = names_list[0]
            sig_name2 = names_list[1]
            sig_name3 = names_list[2]
            sig_name4 = names_list[3]
            sig_name5 = names_list[4]
            sig_name6 = names_list[5]
            sig_name7 = names_list[6]
            label_name1_2 = names_list[7]
            label_name2_3 = names_list[8]
            label_name3_4 = names_list[9]
            label_name4_5 = names_list[10]
            label_name5_6 = names_list[11]
            label_name6_7 = names_list[12].strip()


            dataset.append([sig_name1, sig_name2,sig_name3,sig_name4,sig_name5,sig_name6,sig_name7,label_name1_2,label_name2_3,label_name3_4,label_name4_5,label_name5_6,label_name6_7])

    else:

        for line in files_list:
            names_list = line.split()
            if len(names_list) < 1: # empty line
                    continue
            sig_name1 = names_list[0]
            sig_name2 = names_list[1]
            sig_name3 = names_list[2]
            sig_name4 = names_list[3]
            sig_name5 = names_list[4]
            sig_name6 = names_list[5]
            sig_name7 = names_list[6].strip()
            dataset.append([sig_name1, sig_name2, sig_name3, sig_name4, sig_name5,sig_name6,sig_name7])


    return dataset






class distribution_maps(data.Dataset):
    """
    Read dataset of kaggle ultrasound nerve segmentation dataset
    https://www.kaggle.com/c/ultrasound-nerve-segmentation
    """

    def __init__(self, root,list_txt,image_width, train=True):
        self.train = train
        self.root = root
        # we cropped the image(the size of each patch, can be divided by 16 )
        self.nCol=image_width
        self.nRow=image_width
        self.data = make_dataset(root, list_txt,train)

    def __getitem__(self, idx):
        if self.train:
            sig_data1, sig_data2,sig_data3, sig_data4, sig_data5,sig_data6,sig_data7, gt_patch1_2,gt_patch2_3,gt_patch3_4,gt_patch4_5,gt_patch5_6,gt_patch6_7 = self.data[idx]
           # print(idx)

            sig1 = read_patch(sig_data1,self.nCol)
            sig2=read_patch(sig_data2,self.nCol)
            sig3 = read_patch(sig_data3, self.nCol)
            sig4 = read_patch(sig_data4, self.nCol)
            sig5 = read_patch(sig_data5, self.nCol)
            sig6 = read_patch(sig_data6, self.nCol)
            sig7 = read_patch(sig_data7, self.nCol)
            gt1_2=  int(gt_patch1_2)
            gt2_3 = int(gt_patch2_3)
            gt3_4 = int(gt_patch3_4)
            gt4_5 = int(gt_patch4_5)
            gt5_6=int(gt_patch5_6)
            gt6_7=int(gt_patch6_7)
            sig=np.stack((sig1,sig2,sig3,sig4,sig5,sig6,sig7))
            sig = np.atleast_2d(sig).astype(np.float32)
            sig = torch.from_numpy(sig).float()
            gt = np.stack((gt1_2, gt2_3, gt3_4, gt4_5,gt5_6,gt6_7))
            gt = np.atleast_1d(gt)
            gt = torch.from_numpy(gt).float()

            return sig, gt
        else:
            sig_data1, sig_data2, sig_data3, sig_data4, sig_data5,sig_data6,sig_data7 = self.data[idx]
            sig1 = read_patch(sig_data1, self.nCol)
            sig2 = read_patch(sig_data2, self.nCol)
            sig3 = read_patch(sig_data3, self.nCol)
            sig4 = read_patch(sig_data4, self.nCol)
            sig5 = read_patch(sig_data5, self.nCol)
            sig6 = read_patch(sig_data6, self.nCol)
            sig7 = read_patch(sig_data7, self.nCol)

            sig = np.stack((sig1, sig2, sig3, sig4, sig5,sig6,sig7))
            sig = np.atleast_2d(sig).astype(np.float32)
            sig = torch.from_numpy(sig).float()


            return sig

    def __len__(self):
        count = len(self.data)

        return count


