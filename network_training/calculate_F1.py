
import argparse
import numpy as np


def calculate_F1(out,label):
    a=(out==label)&(label==1)
    TP=sum(a)
    number_P=sum(label)
    number_P_out=sum(out)
    FP=number_P_out-TP
    FN=number_P-TP
    if (TP + FP) == 0:
        p = 0
    else:
        p = float(TP / (TP + FP))
    if (TP + FN) == 0:
        r = 0
    else:
        r = float(TP / (TP + FN))
    if (TP + 0.5 * FP + 0.5 * FN) == 0:
        F1 = 0
    else:
        F1 = float(TP / (TP + 0.5 * FP + 0.5 * FN))

    print("number of positive: %d"%(number_P))
    print("number of TP: %d"%TP)
    print("number of FP: %d"%FP)
    print("F1 score is %f"%F1)





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='input image list')
    parser.add_argument('--label', help="input label list")
    #parser.add_argument('--inf',action='store_true',default=False,help="whether making the inference pair")
    args = parser.parse_args()
    out=np.loadtxt(args.output)
    label=np.loadtxt(args.label)
    if len(out)!=len(label):
        print ("two files have different lenght")
        assert False

    calculate_F1(out[:,0],label)
