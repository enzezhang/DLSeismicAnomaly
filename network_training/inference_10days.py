import argparse
import os
import numpy as np
from tqdm import tqdm


#from dataloaders import make_data_loader
from model.resnet_Enze import *
import torch.nn as nn

from torch.autograd import Variable
import time

import sys
HOME = os.path.expanduser('~')
basicCodes_path = HOME + '/zez_code'
sys.path.append(basicCodes_path)
from dataset_EQ_distribution_10days import *
import parameters


parser = argparse.ArgumentParser()
parser.add_argument('dataroot', help='path to test dataset ')
parser.add_argument('para', help='path to the parameter file')
parser.add_argument('list', help='path to the list file')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')

parser.add_argument('--cuda', action='store_true', help='enables cuda')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


args = parser.parse_args()
print(args)

parameters.set_saved_parafile_path(args.para)

crop_width = parameters.get_digit_parameters("", "crop_width", None, 'int')

dataset = distribution_maps(args.dataroot, args.list, crop_width,train=False)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                           num_workers=args.workers, shuffle=False)


#model = resnet18(num_img=10,num_classes=9)
model = resnet34(num_img=10,num_classes=9)
#model=resnet101(num_img=10,num_classes=9)
if args.cuda:
    model.cuda()

if args.resume:
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        #if args.cuda:
            #model.module.load_state_dict(checkpoint['state_dict'])
        #else:
        model.load_state_dict(checkpoint['state_dict'])
        best_pred = checkpoint['best_pred']

        print("=> loaded checkpoint '{}' (epoch {}) with best F1 {}"
              .format(args.resume, checkpoint['epoch'],best_pred))




    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        assert False
else:
    print("Please input the check point files")

model.eval()
pos_weight=np.atleast_1d(3)
pos_weight = torch.from_numpy(pos_weight).float()
criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
if args.cuda:
            criterion=criterion.cuda()
patch_number = len(train_loader)
start_time=time.time()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
TP=0
FN=0
FP=0
P_ground_truth=0
N_ground_truth=0
test_loss=0
output=np.zeros(len(dataset)+9)
weight=np.zeros(len(dataset)+9)
weight[0]=1
for i, (x) in enumerate(train_loader):
    if args.cuda:
        x = x.cuda()
    with torch.no_grad():
        y = model(Variable(x))
        y=np.round(sigmoid(y.cpu().numpy()))
        y=y.flatten()
    output[i+1:i+10]+=np.array(y)
    weight[i+1:i+10]+=np.ones(9)

temp1=os.path.basename(args.resume)
temp1=('.').join(temp1.split('.')[:-1])
temp2=os.path.basename(args.list)
temp2=('.').join(temp2.split('.')[:-1])
filename=temp1+'_'+temp2+'.txt'
for i in range(0,len(output)):
    with open(filename, 'a') as log:
        temp=float(output[i])/float(weight[i])
        temp2=(temp>0.4)
        out_massage=("%d %d"%(temp2,output[i]))
        log.writelines(out_massage + '\n')





#
# TP = float(TP)
# FP = float(FP)
# FN = float(FN)
# if (TP + FP) == 0:
#     p = 0
# else:
#     p = float(TP / (TP + FP))
# if (TP + FN) == 0:
#     r = 0
# else:
#     r = float(TP / (TP + FN))
# F1 = float(TP / (TP + 0.5 * FP + 0.5 * FN))
# print("test loss: %f"%(test_loss/len(train_loader)))
# print("number of P:%d  number of N: %d  TP:%d   FP:%d   FN:%d  precision: %.2f    recall: %.2f    F1 score: %.2f" % (P_ground_truth,N_ground_truth,TP, FP, FN, p, r, F1))
# end_time=time.time()
# print("Inference time used is %f" %(end_time-start_time))


