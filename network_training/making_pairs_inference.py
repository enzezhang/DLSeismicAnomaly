
import argparse




def making_pairs(image,label):
    flag=0
    for i in range(0,len(image)-1):
        if label[i+1] == '0':
            print("%s %s %s" %(image[i], image[i + 1], label[i + 1]))
        else:
            for l in range(flag,i+1):
                print("%s %s %s" %(image[l], image[i + 1], label[i + 1]))
                flag=i+2


def making_pairs_inference(image):
    for i in range(0,len(image)-1):
        print("%s %s"%(image[i],image[i+1]))







if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='input image list')

    args = parser.parse_args()
    with open(args.image) as f_image:
        image_line=f_image.readlines()
    if len(image_line) < 1:
        print("error, no file name in the %s" % args.image)
        assert False

    image_list=[]
    for line in image_line:
        image_list.append(line.strip())
    making_pairs_inference(image_list)

