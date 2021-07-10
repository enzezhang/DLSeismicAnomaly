import argparse
import matplotlib.pyplot as plt


parser=argparse.ArgumentParser()
parser.add_argument('--log', type=str, help='absolute path of logfile')


parser.add_argument('--image', type=str, help='absolute path of image file')

parser.add_argument('start',type=int, help='start epoch')
args = parser.parse_args()


def read_loss(log):
    with open(log, 'r') as f:
        data = f.readlines()
        epoch=[]
        loss=[]
        for line in data:
            line = line.strip('\n')
            extract=line.split(" ")
            if (extract[0]=='[Epoch:'):
                if (extract[2]=='numImages:'):
                    print(extract)
                    epoch.append(float(extract[1].strip(',')))
            if (extract[0] == 'Loss:'):
                loss.append(float(extract[1]))
    return epoch, loss

def read_val(log):
    with open(log, 'r') as f:
        data = f.readlines()
        loss=[]
        for line in data:
            line = line.strip('\n')
            extract=line.split(" ")
            if (len(extract)>1):
                if (extract[1]=='Loss:'):
                    print(extract)
                    print(extract[2])
                    loss.append(float(extract[2]))

    return loss

def read_iteration(log):
    with open(log, 'r') as f:
        data = f.readlines()
        loss=[]
        iteration=[]
        for line in data:
            line = line.strip('\n')
            extract=line.split(" ")
            if (len(extract)>3):
                if (extract[3]=='iteration'):
                    loss.append(float(extract[6]))
                    iteration.append(int(extract[4]))

    return iteration,loss





def main(log,image,start):
    [epoch,loss]=read_loss(log)
    val=read_val(log)
    print(len(epoch),len(loss),len(val))

    # epoch,loss=read_iteration(log)
    print (len(epoch))
    #print(epoch[0])
    plt.figure()
    plt.plot(epoch[start:], loss[start:])
    plt.plot(epoch[start:], val[start:])

    plt.legend(('train error','validation error'),loc='upper right')
    plt.savefig(image)
    #plt.show()






if (__name__ == '__main__'):
    print(args)
    main(str(args.log),args.image,args.start)
