import argparse
import sys
import os
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='output file')
    parser.add_argument('--list', help="image list")
    args = parser.parse_args()
    with open(args.output,"r") as f:
        output=f.readlines()
    with open(args.list,"r") as f:
        list=f.readlines()
    if (len(output)!=len(list)):
        print("the length of the output and image list are not identical, please double check")
        sys.exit()
    else:
        for i in range (0,len(output)):
            line_out=output[i].strip('\n')
            label=int(line_out[0])
            if (label):
                line_list=list[i].strip('\n')
                file_name=os.path.split(line_list)
                date=file_name[1].split("_lola",1)[0]
                print(date)

