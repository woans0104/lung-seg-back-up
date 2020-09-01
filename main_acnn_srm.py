
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import shutil
import pickle

import argparse

import ipdb

import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn

from medpy.metric import binary

import dataloader as loader
from utils import *
from predict import main_test
from model import *
from losses import DiceLoss,ClDice

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
parser.add_argument('--arch-seg', default='unet', type=str)
parser.add_argument('--arch-ae', default='ae', type=str)
parser.add_argument('--load-ae-path', default='load_model_path', type=str)
parser.add_argument('--embedding-alpha', default=1, type=float)
parser.add_argument('--embedding-beta', default=1, type=float)


# arguments for optim & loss
parser.add_argument('--optim',default='sgd',choices=['adam','sgd'],type=str)
parser.add_argument('--weight-decay',default=5e-4,type=float)

parser.add_argument('--seg-loss-function',default='bce',type=str)
parser.add_argument('--ae-loss-function',default='bce',type=str)

parser.add_argument('--lr', default=0.1, type=float, help='initial-lr')
parser.add_argument('--lr-schedule', default=[100,120], nargs='+', type=int)


# arguments for test mode
parser.add_argument('--file-name', default='result_all', type=str)
parser.add_argument('--test_mode',default=True,type=str2bool)


args = parser.parse_args()


def main():
    # save input stats for later use

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
    train_loader_source,test_loader_source = loader.get_loader(server=args.server,dataset=source_dataset,train_size=1,
                                            aug_mode=args.aug_mode,aug_range=args.aug_range,
                                            batch_size=args.batch_size,work_dir=work_dir)

    train_loader_target1, _ = loader.get_loader(server=args.server, dataset=target_dataset1, train_size=1,
                                            aug_mode=False, aug_range=args.aug_range,batch_size=1, work_dir=work_dir)
    train_loader_target2, _ = loader.get_loader(server=args.server, dataset=target_dataset2, train_size=1,
                                            aug_mode=False, aug_range=args.aug_range,batch_size=1, work_dir=work_dir)


    test_data_li = [test_loader_source, train_loader_target1, train_loader_target2]


    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))


    # 2.model_select
    model_seg, _ = utils.select_model(args.arch_seg)
    model_seg = nn.DataParallel(model_seg).cuda()

    # load_model
    model_ae,_ = utils.select_model(args.arch_ae)
    model_ae = nn.DataParallel(model_ae).cuda()

    checkpoint_path = os.path.join(args.load_ae_path, 'model_best.pth')
    state = torch.load(checkpoint_path)
    model_ae.load_state_dict(state['state_dict'])

    cudnn.benchmark = True

    # 3.optim

    if args.optim == 'adam':
        optimizer_seg = torch.optim.Adam(model_seg.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer_seg = torch.optim.SGD(model_seg.parameters(), lr=args.lr,
                                        momentum=0.9, weight_decay=args.weight_decay)

    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_seg, milestones=lr_schedule[:-1], gamma=0.1)


    # 4.loss

    criterion_seg = utils.select_loss(args.seg_loss_function)
    criterion_ae = utils.select_loss(args.ae_loss_function)


#####################################################################################

    # train

    send_slack_message('#jm_private', '{} : starting_training'.format(args.exp))
    best_iou = 0
    try:
        if args.train_mode:
            for epoch in range(lr_schedule[-1]):

                train(model_seg=model_seg,model_ae=model_ae,train_loader=train_loader,epoch=epoch,
                      criterion_seg=criterion_seg,criterion_ae=criterion_ae,optimizer_seg= optimizer_seg,
                      logger=trn_logger, sublogger=trn_raw_logger)

                iou = validate(model=model_seg,val_loader=val_loader,epoch= epoch,criterion=criterion_seg,
                               logger=val_logger)
                print('validation_result **************************************************************')

                lr_scheduler.step()

                if args.val_size ==0:
                    is_best = 1
                else:
                    is_best = iou > best_iou
                best_iou = max(iou, best_iou)
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': my_net.state_dict(),
                                 'optimizer': gen_optimizer.state_dict()},
                                is_best,work_dir,filename='checkpoint.pth')

        print("train end")
    except RuntimeError as e:
        send_slack_message('#jm_private',
                       '-----------------------------------  error train : send to message JM  '
                       '& Please send a kakao talk ----------------------------------------- '
                       '\n error message : {}'.format(e))

        import ipdb
        ipdb.set_trace()


    draw_curve(work_dir, trn_logger, val_logger)
    send_slack_message('#jm_private', '{} : end_training'.format(args.exp))

    # here is load model for last pth
    utils.check_best_pth(work_dir)

    # validation
    if args.test_mode:
        print('Test mode ...')
        main_test(model=model_seg, test_loader=test_data_li, args=args)





def train(model_seg, model_ae, train_loader, epoch, criterion_seg, criterion_ae, optimizer_seg, logger, sublogger):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    embedding_losses = AverageMeter()
    recon_losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()


    model_seg.train()
    model_ae.eval()
    end = time.time()

    for i, (input, target,_,_) in enumerate(train_loader):

        data_time.update(time.time() - end)
        input, target = input.cuda(), target.cuda()


        output_seg,bottom_seg = model_seg(input)
        loss_seg = criterion_seg(output_seg, target)


        # training model
        if args.arch_seg == 'SRM':

            output_srm, bottom_srm = model_ae(output_seg) # input seg-output
            _, bottom_ae = model_ae(target) # input gt


            bottom_ae = F.sigmoid(bottom_ae).cuda()
            loss_embedding = select_loss_ae(args.ae_loss_function, criterion_ae, bottom_srm, bottom_ae)

            loss_recon = DiceLoss().cuda()
            loss_recon = loss_recon(output_srm, target)

            total_loss = (loss_seg) + (float(args.embedding_alpha) * loss_embedding)\
                         + (float(args.embedding_beta) * loss_recon)


        elif args.arch_seg == 'ACNN':

            _, bottom_acnn = model_ae(output_seg)
            _, bottom_ae = model_ae(target)
            loss_embedding = utls.embedding_loss(args.ae_loss_function, criterion_ae, bottom_acnn, bottom_ae)

            loss_embedding = float(args.embedding_alpha) * loss_embedding
            total_loss = (loss_seg) + (loss_embedding)

        else:
            print('Not training')
            import ipdb;ipdb.set_trace()


        print('loss_seg : ', loss_seg)
        print('alpha * loss_embedding : ', loss_embedding)
        if args.arch_seg == 'SRM':
            print('beta * loss_recon : ', loss_recon)
        print('Total-loss : ', total_loss)


        iou, dice = utils.performance(output_seg, target,dist_con=False)
        losses.update(total_loss.item(), input.size(0))
        if args.arch_seg == 'SRM':
            recon_losses.update(loss_recon.item(), input.size(0))
        embedding_losses.update(loss_embedding.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices.update(dice, input.size(0))


        optimizer_seg.zero_grad()
        total_loss.backward()
        optimizer_seg.step()

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

    if args.arch_seg == 'SRM':
        logger.write([epoch, losses.avg,embedding_losses.avg,recon_losses.avg, ious.avg, dices.avg])
    else:
        logger.write([epoch, losses.avg, embedding_losses.avg, ious.avg, dices.avg])


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

            output,_ = model(input)
            loss = criterion(output, target)


            iou, dice = utils.performance(output_seg, target, dist_con=False)

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