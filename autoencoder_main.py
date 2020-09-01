

import os
import time

import shutil
import pickle

import argparse

import ipdb

import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import matplotlib.pyplot as plt

from medpy.metric import binary

import dataloader as loader
from utils import Logger, AverageMeter, save_checkpoint ,draw_curve ,send_slack_message,str2bool
from autoencoder_pred import main_test
from model import *
from losses import DiceLoss,tversky_loss, NLL_OHEM,ClDice


parser = argparse.ArgumentParser()


# arguments for dataset
parser.add_argument('--server', default='server_B')
parser.add_argument('--exp',default="test4", type=str)
parser.add_argument('--train_mode',default=True,type=str2bool)

parser.add_argument('--source-dataset',default='JSRT',help='JSRT_dataset,MC_dataset,SH_dataset')

parser.add_argument('--train-size',default=0.7,type=float)
parser.add_argument('--val-size',default=0,type=float)

parser.add_argument('--batch-size',default=8,type=int)
parser.add_argument('--aug-mode',default=False,type=str2bool)
parser.add_argument('--aug-range',default='aug6', type=str)


# arguments for model
parser.add_argument('--arch', default='unet', type=str)
parser.add_argument('--denoising',default=False,type=str2bool)
parser.add_argument('--salt-prob', default=0.1, type=float)

# arguments for optim & loss
parser.add_argument('--optim',default='sgd',choices=['adam','sgd'],type=str)
parser.add_argument('--weight-decay',default=1e-4,type=float)

parser.add_argument('--loss-function',default='bce',type=str)
parser.add_argument('--lr',default=0.1,type=float,help='initial-lr')
parser.add_argument('--lr-schedule', default=[100,120], nargs='+', type=int)

# arguments for test mode
parser.add_argument('--file-name', default='result_all', type=str)
parser.add_argument('--test-mode',default=True,type=str2bool)

args = parser.parse_args()


def main():

    if args.server == 'server_A':
        work_dir = os.path.join('/data1/JM/lung_segmentation', args.exp)
        print(work_dir)
    elif args.server == 'server_B':
        work_dir = os.path.join('/data1/workspace/JM_gen/lung_seg', args.exp)
        print(work_dir)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # copy this file to work dir to keep training configuration
    shutil.copy(__file__, os.path.join(work_dir, 'main.py'))
    with open(os.path.join(work_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    source_dataset, target_dataset1, target_dataset2 = loader.dataset_condition(args.source_dataset)

    # 1.load_dataset
    train_loader_source, test_loader_source = loader.get_loader(server=args.server, dataset=source_dataset,
                                                                train_size=1,
                                                                aug_mode=args.aug_mode, aug_range=args.aug_range,
                                                                batch_size=args.batch_size, work_dir=work_dir)

    train_loader_target1, _ = loader.get_loader(server=args.server, dataset=target_dataset1, train_size=1,
                                                aug_mode=False, aug_range=args.aug_range, batch_size=1,
                                                work_dir=work_dir)
    train_loader_target2, _ = loader.get_loader(server=args.server, dataset=target_dataset2, train_size=1,
                                                aug_mode=False, aug_range=args.aug_range, batch_size=1,
                                                work_dir=work_dir)

    test_data_li = [test_loader_source, train_loader_target1, train_loader_target2]

    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))

    # 2.model_select
    model_ae = utils.select_model(args.arch)

    # 3.gpu select
    model_ae = nn.DataParallel(model_ae).cuda()
    cudnn.benchmark = True

    # 4.optim
    if args.optim == 'adam':
        optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer_ae = torch.optim.SGD(model_ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_ae, milestones=lr_schedule[:-1])

    # 5.loss
    criterion = utils.select_loss(args.loss_function)


#####################################################################################

    # train

    utils.send_slack_message('#jm_private', '{} : starting_training'.format(args.exp))
    best_iou = 0
    try:
        if args.train_mode:
            for epoch in range(lr_schedule[-1]):

                train(model=model_ae, train_loader=train_loader, epoch=epoch, criterion=criterion,
                      optimizer=optimizer_ae, logger=trn_logger, sublogger=trn_raw_logger)
                iou = validate(model=model_ae, val_loader=val_loader, epoch=epoch, criterion=criterion,
                               logger=val_logger, work_dir=work_dir, save_fig=True,
                               work_dir_name='{}_visualize_per_epoch'.format(args.source_dataset))
                print('validation_result **************************************************************')

                lr_scheduler.step()

                if args.val_size == 0:
                    is_best = 1
                else:
                    is_best = iou > best_iou

                best_iou = max(iou, best_iou)
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model_ae.state_dict(),
                                 'optimizer': gen_optimizer.state_dict()},
                                is_best, work_dir, filename='checkpoint.pth')

            print("train end")
    except RuntimeError as e:
        send_slack_message('#jm_private',
                           '-----------------------------------  error train : send to message JM '
                           '& Please send a kakao talk ----------------------------------------- \n error message : {}'
                           .format(e))
        import ipdb
        ipdb.set_trace()

    utils.draw_curve(work_dir, trn_logger, val_logger)
    utils.send_slack_message('#jm_private', '{} : end_training'.format(args.exp))
    # here is load model for last pth
    utils.check_best_pth(work_dir)

    # validation
    if args.test_mode:
        print('Test mode ...')
        main_test(model=model_ae,test_loader=test_data_list, args=args)


def train(model, train_loader,epoch, criterion, optimizer, logger, sublogger):
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

        if args.denoising == True:
            noisy_batch_input = utils.make_noise_input(target)

            output, _ = model(noisy_batch_input)
            loss = criterion(output, target)


        else:
            #cae
            output, core = model(target)
            loss = criterion(output, target)


        iou, dice = utils.performance(output, target,dist_con=False)
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
            # print("pix_count : ",pix_count)

    logger.write([epoch, losses.avg, ious.avg, dices.avg])




def validate(model, val_loader, epoch, criterion, logger, work_dir, save_fig=False, work_dir_name=False):

    batch_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target,ori_img,image_name) in enumerate(val_loader):

            input = input.cuda()
            target = target.cuda()

            output,_ = model(target)

            loss = criterion(output, target)

            iou, dice = utils.performance(output, target, dist_con=False)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))

            if save_fig:
                if i % 10 == 0:
                    utils.save_fig(str(epoch), ori_img, target, output, iou, work_dir,work_dir_name, image_name[0])

            batch_time.update(time.time() - end)
            end = time.time()

    print(' * IoU {ious.avg:.3f}({ious.std:.3f}) Dice {dices.avg:.3f}({dices.std:.3f})'.format(
           ious=ious, dices=dices))

    logger.write([epoch, losses.avg, ious.avg, dices.avg])


    return ious.avg







if __name__ == '__main__':
    main()