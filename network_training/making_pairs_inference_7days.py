
import argparse






def making_pairs_inf(image):
    for i in range(0,len(image)-6):
        print('%s %s %s %s %s %s %s' % (image[i], image[i + 1],image[i + 2],image[i + 3],image[i + 4],image[i + 5],image[i + 6]))





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='input image list')

    #parser.add_argument('--inf',action='store_true',default=False,help="whether making the inference pair")
    args = parser.parse_args()
    with open(args.image) as f_image:
        image_line=f_image.readlines()
    if len(image_line) < 1:
        print("error, no file name in the %s" % args.image)
        assert False

    image_list=[]

    for line in image_line:
        image_list.append(line.strip())

    making_pairs_inf(image_list)
    #else:
       # making_pairs(image_list,label_list)
