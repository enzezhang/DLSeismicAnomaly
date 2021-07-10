import argparse
import os
import numpy as np
# from tqdm import tqdm
import time
import torch
#from dataloaders import make_data_loader


# from utils.loss import SegmentationLosses
# from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
# from utils.summaries import TensorboardSummary
from model.sync_batchnorm.replicate import patch_replication_callback
from utils.metrics import Evaluator
#from pytorchtools import EarlyStopping
#early_stopping = EarlyStopping(patience=5, verbose=True)
import sys
basicCodes_path = '/home/zez/test_deep_learning/u_net/Unet_pytorch-master_Dec5'
sys.path.append(basicCodes_path)

from dataset_EQ_distribution_10days import *
import parameters
from model.resnet_Enze import *
from torchsummary import summary

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary

        # self.summary = TensorboardSummary(self.saver.experiment_dir)
        #self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}

        parameters.set_saved_parafile_path(args.para)





        crop_width = parameters.get_digit_parameters("", "crop_width", None, 'int')


        dataset = distribution_maps(args.dataroot, args.list, crop_width)

        # train_loader_test = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
        #                                            num_workers=args.workers, shuffle=True,drop_last=True)
        self.train_length = int(len(dataset) * 0.9)
        self.validation_length = len(dataset) - self.train_length
        [train_data,validation_data]=torch.utils.data.random_split(dataset, (self.train_length, self.validation_length))
        self.train_loader=torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size,
                                                        num_workers=args.workers, shuffle=True)
        print("the train loader length is %d and val_length is %d" % (self.train_length, self.validation_length ))
        print("the train loader dataset length is %d and val loader dataset length is %d" % (len(self.train_loader), len(self.val_loader)))

        #self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network



        #model = resnet18(num_img=10,num_classes=9)
        model=resnet34(num_img=10,num_classes=9)
        #model=resnet101(num_img=10,num_classes=9)
        #model= resnet10(num_classes=4)
        # print(model)

        # train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                 {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay, nesterov=args.nesterov)
        #optimizer=torch.optim.Adamax(model.parameters(),lr=args.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.01)

        pos_weight=np.atleast_1d(1)
        pos_weight = torch.from_numpy(pos_weight).float()
        self.criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if args.cuda:
            self.criterion=self.criterion.cuda()


        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(2)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            #self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = torch.nn.DataParallel(self.model)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 999999999
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {}) with best loss {}"
                  .format(args.resume, checkpoint['epoch'], checkpoint['best_pred']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
            self.best_pred=999999999

    def training(self, epoch):
        train_start_time=time.time()
        train_loss = 0.0
        self.model.train()
        #tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        TP=0
        FP=0
        FN=0
        P_ground_truth=0
        o_truth=0
        for i, (x, y) in enumerate(self.train_loader):
            start_time = time.time()
            image, target = x, y
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            end_time = time.time()
            out_detech=output.detach()
            target_numpy=target.cpu().numpy()
            out_numpy=np.round(sigmoid(out_detech.cpu().numpy()))
            P_ground_truth_temp=0
            TP_temp=0
            o_truth_temp=0
            for l in range(0,len(target_numpy)):
                if (target_numpy[l,:].any()):
                    P_ground_truth_temp+=1
                    if ((out_numpy[l,:]==target_numpy[l,:]).all()):
                        TP_temp+=1
                if (out_numpy[l,:].any()):
                    o_truth_temp+=1



            P_ground_truth+=P_ground_truth_temp
            TP+=TP_temp
            FP+=(o_truth_temp-TP_temp)
            FN+=(P_ground_truth_temp-TP_temp)

            print('[The loss for iteration %d is %.3f and the time used is %.3f]' % (
            i + num_img_tr * epoch, loss.item(), end_time - start_time))

        train_end_time=time.time()
        TP = float(TP)
        FP = float(FP)
        FN = float(FN)
        if (TP + FP) == 0:
            p = 0
        else:
            p = float(TP / (TP + FP))
        if (TP + FN) == 0:
            r = 0
        else:
            r = float(TP / (TP + FN))
        if (TP + 0.5 * FP + 0.5 * FN) == 0:
            F1=0
        else:
            F1 = float(TP / (TP + 0.5 * FP + 0.5 * FN))
        print('[Epoch: %d,training time used is %.3f]' % (epoch, train_end_time - train_start_time))

        print('Loss: %.3f' % (train_loss/len(self.train_loader)))
        print("number of P:%d   TP:%d   FP:%d   FN:%d  precision: %.2f    recall: %.2f    F1 score: %.2f" % (P_ground_truth,TP, FP, FN, p, r, F1))
        name=self.args.checkname+".txt"
        file_name = os.path.join(self.args.dataroot, name)
        with open(file_name, 'a') as log:
            out_massage = '[Epoch: %d, numImages: %5d]' % (epoch, self.train_length)
            log.writelines(out_massage + '\n')
            out_massage = 'Loss: %.3f' % (train_loss / len(self.train_loader))
            log.writelines(out_massage + '\n')
            out_message = 'number of P:%d   TP:%d   FP:%d   FN:%d  precision: %.2f    recall: %.2f    F1 score: %.2f' % (P_ground_truth,
                TP, FP, FN, float(p), float(r), F1)
            log.writelines(out_message + '\n')


        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        val_start_time=time.time()
        self.model.eval()
        self.evaluator.reset()
        num_val=len(self.val_loader)
        #tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        FN=0
        TP=0
        FP=0
        P_ground_truth=0
        for i, (x, y) in enumerate(self.val_loader):
            image, target = x,y
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)

            target_numpy = target.cpu().numpy()
            out_numpy = np.round(sigmoid(output.cpu().numpy()))
            P_ground_truth_temp = 0
            TP_temp = 0
            o_truth_temp = 0
            for l in range(0, len(target_numpy)):
                if (target_numpy[l, :].any()):
                    P_ground_truth_temp += 1
                    if ((out_numpy[l, :] == target_numpy[l, :]).all()):
                        TP_temp += 1
                if (out_numpy[l, :].any()):
                    o_truth_temp += 1

            P_ground_truth += P_ground_truth_temp
            TP += TP_temp
            FP += (o_truth_temp - TP_temp)
            FN += (P_ground_truth_temp - TP_temp)
            test_loss += loss.item()
            #tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            # pred = output.data.cpu().numpy()
            # target = target.cpu().numpy()
            # #pred = np.argmax(pred, axis=1)
            # # Add batch sample into evaluator


        print('------------Validation------------')
        val_end_time=time.time()
        print('[Epoch: %d, numImages: %5d, validation time used is %.3f]' % (epoch,self.validation_length,val_end_time-val_start_time))
        # print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        validation_loss=test_loss/len((self.val_loader))
        TP=float(TP)
        FP=float(FP)
        FN=float(FN)
        if (TP+FP)==0:
            p=0
        else:
            p = float(TP / (TP + FP))
        if (TP+FN)==0:
            r=0
        else:
            r = float(TP / (TP + FN))
        if (TP + 0.5 * FP + 0.5 * FN) == 0:
            F1=0
        else:
            F1 = float(TP / (TP + 0.5 * FP + 0.5 * FN))
        print("number of positive: %d    TP:%d   FP:%d   FN:%d  precision: %.2f    recall: %.2f    F1 score: %.2f"%(P_ground_truth,TP,FP,FN,p,r,F1))
        print('Validation Loss: %.3f' % (validation_loss))
        name=self.args.checkname+".txt"

        file_name = os.path.join(self.args.dataroot, name)

        with open(file_name, 'a') as log:
            out_message = 'Validation:'
            log.writelines(out_message + '\n')
            out_message = 'Validation Loss: %.3f' % (validation_loss)
            log.writelines(out_message + '\n')
            out_message = 'number of positive: %d     TP:%d   FP:%d   FN:%d  precision: %.2f    recall: %.2f    F1 score: %.4f' % (
                P_ground_truth,TP, FP, FN, p, r, F1)

            log.writelines(out_message + '\n')
            out_message = 'Best valication loss is %.3f'%(self.best_pred)
            log.writelines(out_message + '\n')

        new_pred = validation_loss


        if new_pred < self.best_pred:
            print('validation loss  is %.3f and lower then previous best %.3f, save model' % (new_pred,self.best_pred))
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best,self.args.checkname)
            return False
        else:
            return True





def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],help='backbone name (default: resnet)')

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')



    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # parser.add_argument('--loss-type', type=str, default='ce',
    #                     choices=['ce', 'focal'],
    #                     help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    # parser.add_argument('--test-batch-size', type=int, default=None,
    #                     metavar='N', help='input batch size for \
    #                             testing (default: auto)')

    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')



    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')

    #related to SDG
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=5, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')

    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--patience', type=int,default=5,help='patience for early stopping')
    parser.add_argument('dataroot', help='path to dataset of kaggle ultrasound nerve segmentation')
    parser.add_argument('para', help='path to the parameter file')
    parser.add_argument('list', help='path to the list file')


    args = parser.parse_args()



    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    # if args.epochs is None:
    #     epoches = {
    #         'coco': 30,
    #         'cityscapes': 200,
    #         'pascal': 50,
    #     }
    #     args.epochs = epoches[args.dataset.lower()]
    #
    # if args.batch_size is None:
    #     args.batch_size = 4 * len(args.gpu_ids)
    #
    # if args.test_batch_size is None:
    #     args.test_batch_size = args.batch_size
    #
    # if args.lr is None:
    #     lrs = {
    #         'coco': 0.1,
    #         'cityscapes': 0.01,
    #         'pascal': 0.007,
    #     }
    #     args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    # delete the seed
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    early_stop_count=0
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            early_stop=trainer.validation(epoch)
            if early_stop:
                early_stop_count += 1
                print ("early stop count is %d"%(early_stop_count))
            else:
                early_stop_count=0
            if early_stop_count > args.patience:
                print("early stopping at epoch %d"%(epoch))
                break

    #trainer.writer.close()

if __name__ == "__main__":
   main()
