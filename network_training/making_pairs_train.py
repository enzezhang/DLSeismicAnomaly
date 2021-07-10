
import argparse




def making_pairs(image,label):
    flag=0
    for i in range(0,len(image)-1):
        if label[i+1] == '0':
            print("%s %s %s" %(image[i], image[i + 1], label[i + 1]))
        else:
            for l in range(flag,i+1):
                print("%s %s %s" %(image[l], image[i + 1], label[i + 1]))
                flag=i+1

def making_pairs_ref(image,label):
    for i in range(0,len(image)-1):
        print("%s %s %s" %(image[i], image[i + 1], label[i + 1]))







if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='input image list')
    parser.add_argument('--label', help="input label list")
    parser.add_argument('--ref',action='store_true',default=False,help="whether making the reference pair")
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
    if args.ref:
        making_pairs_ref(image_list,label_list)
    else:
        making_pairs(image_list,label_list)
