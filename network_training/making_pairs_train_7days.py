
import argparse
import numpy as np
import random


def making_pairs(image,label):
    for i in range(0,len(image)-6):
        temp=np.array(label[i+1:i+6])
        if np.all(temp=='0'):
            a=random.randint(1,3)
            if a==1 or a==2:
                print("%s %s %s %s %s %s %s %s %s %s %s %s %s" %(image[i], image[i + 1],image[i + 2],image[i + 3],image[i + 4],image[i + 5],image[i + 6], label[i + 1], label[i + 2], label[i + 3], label[i + 4],label[i + 5],label[i + 6]))
        else:
            print("%s %s %s %s %s %s %s %s %s %s %s %s %s" % (
            image[i], image[i + 1], image[i + 2], image[i + 3], image[i + 4], image[i + 5], image[i + 6], label[i + 1],
            label[i + 2], label[i + 3], label[i + 4], label[i + 5],label[i+6]))



#
# def making_pairs_inf(image,label):
#     for i in range(0,len(image)-4):
#         print("%s %s %s %s %s" %(image[i], image[i + 1],image[i + 2],image[i + 3],image[i + 4]))





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='input image list')
    parser.add_argument('--label', help="input label list")
    #parser.add_argument('--inf',action='store_true',default=False,help="whether making the inference pair")
    args = parser.parse_args()
    with open(args.image) as f_image:
        image_line=f_image.readlines()
    if len(image_line) < 1:
        print("error, no file name in the %s" % args.image)
        assert False

    with open (args.label) as f_label:
        label_line=f_label.readlines()
    if len(label_line) < 1:
        print("error, no label in the %s"% args.lable)
        assert False
    image_list=[]
    label_list=[]
    for line in image_line:
        image_list.append(line.strip())
    label_list=[]
    for line2 in label_line:
        label_list.append(line2.strip())
    #if args.ref:
    making_pairs(image_list,label_list)
    #else:
       # making_pairs(image_list,label_list)
