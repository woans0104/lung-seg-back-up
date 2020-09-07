

import os
import time

import shutil
import pickle

import argparse

import ipdb

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from medpy.metric import binary

import dataloader as loader
from utils import *
import torch.optim as optim
from predict import main_test
from adamp import AdamP


parser = argparse.ArgumentParser()


# arguments for dataset
parser.add_argument('--server', default='server_B',type=str)
parser.add_argument('--exp',default='test4',type=str)
parser.add_argument('--train-mode',default=True,type=str2bool)

parser.add_argument('--source-dataset',default='JSRT',help='JSRT_dataset,MC_dataset,SH_dataset')

parser.add_argument('--train-size',default=0.7,type=float)
parser.add_argument('--val-size',default=0,type=float)

parser.add_argument('--batch-size',default=8,type=int)
parser.add_argument('--aug-mode',default=False,type=str2bool)
parser.add_argument('--aug-range',default='aug6', type=str)

# arguments for model
parser.add_argument('--arch', default='unet', type=str)

parser.add_argument('--nomalize-con', default='gn', type=str)
parser.add_argument('--affine', default=True,type=str2bool)
parser.add_argument('--group-channel', default=1, type=int)
parser.add_argument('--weight-std', default=False,type=str2bool)


# arguments for optim & loss
parser.add_argument('--optim',default='sgd',choices=['adam','adamp','sgd'],type=str)
parser.add_argument('--weight-decay',default=5e-4,type=float)

parser.add_argument('--loss-function',default='bce',type=str)
parser.add_argument('--lr',default=0.1,type=float,help='initial-lr')
parser.add_argument('--lr-schedule', default=[100,120], nargs='+', type=int)

# arguments for test mode
parser.add_argument('--file-name', default='result_all', type=str)
parser.add_argument('--test-mode',default=True,type=str2bool)



args = parser.parse_args()





def main():
    # save input stats for later use

    if args.server =='server_A':
        work_dir = os.path.join('/data1/JM/lung_segmentation', args.exp)
        print(work_dir)
    elif args.server =='server_B':
        work_dir = os.path.join('/data1/workspace/JM_gen/lung-seg-back-up', args.exp)
        print(work_dir)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # copy this file to work dir to keep training configuration
    shutil.copy(__file__, os.path.join(work_dir, 'main.py'))
    with open(os.path.join(work_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    source_dataset, target_dataset1, target_dataset2 = loader.dataset_condition(args.source_dataset)



    # 1.load_dataset
    train_loader_source,test_loader_source = loader.get_loader(server=args.server,dataset=source_dataset,
                                                               train_size=args.train_size,aug_mode=args.aug_mode,
                                                               aug_range=args.aug_range,batch_size=args.batch_size,
                                                               work_dir=work_dir)

    train_loader_target1, _ = loader.get_loader(server=args.server, dataset=target_dataset1, train_size=1,
                                                aug_mode=False, aug_range=args.aug_range,batch_size=1, work_dir=work_dir)

    train_loader_target2, _ = loader.get_loader(server=args.server, dataset=target_dataset2, train_size=1,
                                                aug_mode=False, aug_range=args.aug_range,batch_size=1, work_dir=work_dir)



    test_data_li = [test_loader_source, train_loader_target1, train_loader_target2]


    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))


    # 2.model_select
    #model_seg = select_model(args.arch)

    if args.arch == 'unet' :
        model_seg = Unet2D(in_shape=(1, 256, 256))
    elif args.arch == 'unet_norm':
        model_seg = Unet2D_norm(in_shape=(1, 256, 256),nomalize_con=args.nomalize_con,affine=args.affine,
                              group_channel=args.group_channel,weight_std=args.weight_std)

    else:
        raise ValueError('Not supported network.')

    model_seg = model_seg.cuda()




    # 3.gpu select
    model_seg = nn.DataParallel(model_seg)
    cudnn.benchmark = True

    # 4.optim

    if args.optim == 'adam':
        optimizer_seg = torch.optim.Adam(model_seg.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamp':
        optimizer_seg = AdamP(model_seg.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer_seg = torch.optim.SGD(model_seg.parameters(), lr=args.lr,
                                        momentum=0.9,weight_decay=args.weight_decay)


    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_seg,milestones=lr_schedule[:-1], gamma=0.1)


    # 5.loss


    if args.loss_function == 'bce':
        criterion = nn.BCELoss()
    elif args.loss_function == 'bce_logit':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_function == 'dice':
        criterion = DiceLoss()
    elif args.loss_function == 'Cldice':
        bce = nn.BCEWithLogitsLoss().cuda()
        dice = DiceLoss().cuda()
        criterion = ClDice(bce, dice, alpha=1, beta=1)

    criterion = criterion.cuda()






#####################################################################################

    # train

    #utils.send_slack_message('#jm_private', '{} : starting_training'.format(args.exp))
    best_iou = 0
    try:
        if args.train_mode:
            for epoch in range(lr_schedule[-1]):

                train(model=model_seg, train_loader=train_loader_source,epoch=epoch,criterion=criterion,
                      optimizer=optimizer_seg,logger=trn_logger,sublogger= trn_raw_logger)

                iou = validate(model=model_seg,val_loader=test_loader_source, epoch= epoch,criterion=criterion,
                               logger=val_logger)
                print('validation_result **************************************************************')

                lr_scheduler.step()

                if args.val_size ==0:
                    is_best = 1
                else:
                    is_best = iou > best_iou
                best_iou = max(iou, best_iou)
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model_seg.state_dict(),
                                 'optimizer': optimizer_seg.state_dict()},
                                is_best,work_dir,filename='checkpoint.pth')

        print("train end")
    except RuntimeError as e:
        print('error message : {}' .format(e))


        import ipdb
        ipdb.set_trace()


    draw_curve(work_dir, trn_logger, val_logger)
    #utils.send_slack_message('#jm_private', '{} : end_training'.format(args.exp))
    # here is load model for last pth
    check_best_pth(work_dir)

    # validation
    if args.test_mode:
        print('Test mode ...')
        main_test(model=model_seg,test_loader=test_data_li, args=args)




def train(model, train_loader, epoch, criterion, optimizer, logger, sublogger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()


    model.train()
    end = time.time()

    for i, (input, target,_,_) in enumerate(train_loader):


        data_time.update(time.time() - end)
        input, target = input.cuda(), target.cuda()


        output, _ = model(input)
        loss = criterion(output, target)

        iou, dice = performance(output, target,dist_con=False)
        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices.update(dice, input.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'IoU {iou.val:.4f} ({iou.avg:.4f})\t'
              'Dice {dice.val:.4f} ({dice.avg:.4f})\t'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses,
            iou=ious, dice=dices))

        if i % 10 == 0:
            sublogger.write([epoch, i, loss.item(), iou, dice])

    logger.write([epoch, losses.avg, ious.avg, dices.avg])



def validate(model, val_loader, epoch, criterion, logger):

    batch_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, ori_img, image_name) in enumerate(val_loader):

            input = input.cuda()
            target = target.cuda()


            output, _ = model(input)

            loss = criterion(output, target)

            iou, dice = performance(output, target,dist_con=False)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))


            batch_time.update(time.time() - end)
            end = time.time()

    print(' * IoU {ious.avg:.3f}({ious.std:.3f}) Dice {dices.avg:.3f}({dices.std:.3f})'.format(
           ious=ious, dices=dices))

    logger.write([epoch, losses.avg, ious.avg, dices.avg])

    return ious.avg





if __name__ == '__main__':
    main()