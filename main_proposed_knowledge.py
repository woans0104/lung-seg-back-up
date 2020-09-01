

import os
import time
import shutil
import pickle
from copy import deepcopy
import argparse
import ipdb

import numpy as np
import matplotlib.pyplot as plt

import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms

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
parser.add_argument('--train-mode',default=True,type=str2bool)

parser.add_argument('--source-dataset',default='JSRT',help='JSRT_dataset,MC_dataset,SH_dataset')

parser.add_argument('--train-size',default=0.7,type=float)
parser.add_argument('--val-size',default=0,type=float)

parser.add_argument('--batch-size',default=8,type=int)
parser.add_argument('--aug-mode',default=False,type=str2bool)
parser.add_argument('--aug-range',default='aug6', type=str)


# arguments for model
parser.add_argument('--arch-seg', default='unet', type=str)
parser.add_argument('--arch-ae', default='ae_v2', type=str)
parser.add_argument('--arch-ae-detach', default=True, type=str2bool)

parser.add_argument('--embedding-alpha', default=1, type=float)
parser.add_argument('--knowledge-alpha', default=1, type=float)
parser.add_argument('--denoising',default=True,type=str2bool)
parser.add_argument('--salt-prob', default=0.3, type=float)

# arguments for optim & loss
parser.add_argument('--optim',default='sgd',choices=['adam','sgd'],type=str)
parser.add_argument('--weight-decay',default=5e-4,type=float)
parser.add_argument('--ae-weight-decay',default=5e-4,type=float)

parser.add_argument('--seg-loss-function',default='bce',type=str)
parser.add_argument('--ae-loss-function',default='bce',type=str)
parser.add_argument('--embedding-loss-function',default='mse',type=str)

parser.add_argument('--lr', default=0.1, type=float, help='initial-lr')
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
    trn_aelogger = Logger(os.path.join(work_dir, 'ae_trnlogger.log'))
    val_aelogger = Logger(os.path.join(work_dir, 'ae_vallogger.log'))

    # 2.model_select
    model_seg, _ = utils.select_model(args.arch_seg)
    model_ae, _ = utils.select_model(args.arch_ae)

    # 3.gpu select
    model_seg = nn.DataParallel(model_seg).cuda()
    model_ae = nn.DataParallel(model_ae).cuda()
    cudnn.benchmark = True


    # 5.optim
    if args.optim == 'adam':

        optimizer_seg = torch.optim.Adam(model_seg.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=args.lr,weight_decay=args.ae_weight_decay)

    elif args.optim == 'sgd':

        optimizer_seg = torch.optim.SGD(model_seg.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_ae = torch.optim.SGD(model_ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)



    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler_seg = optim.lr_scheduler.MultiStepLR(optimizer_seg,milestones=lr_schedule[:-1],gamma=0.1)
    lr_scheduler_ae = optim.lr_scheduler.MultiStepLR(optimizer_ae,milestones=lr_schedule[:-1],gamma=0.1)


    # 5.loss

    criterion_seg = utils.select_loss(args.seg_loss_function)
    criterion_ae = utils.select_loss(args.ae_loss_function)
    criterion_embedding = utils.select_loss(args.embedding_loss_function)


#####################################################################################

    # train

    send_slack_message('#jm_private', '{} : starting_training'.format(args.exp))
    best_iou = 0
    try:
        if args.train_mode:
            for epoch in range(lr_schedule[-1]):


                train(model_seg =model_seg, model_ae = model_ae,train_loader=train_loader, epoch= epoch,
                      criterion_seg=criterion_seg, criterion_ae=criterion_ae, criterion_embedding=criterion_embedding,
                      optimizer_seg=optimizer_seg,optimizer_ae=optimizer_ae,
                      logger=trn_logger, sublogger=trn_raw_logger,trn_aelogger= trn_aelogger)


                iou = validate(model=model_seg, val_loader= val_loader,epoch= epoch, criterion = criterion_seg,
                               logger= val_logger)


                print('validation result **************************************************************')


                lr_scheduler_seg.step()
                lr_scheduler_ae.step()

                if args.val_size == 0:
                    is_best = 1
                else:
                    is_best = iou > best_iou
                best_iou = max(iou, best_iou)
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': my_net.state_dict(),
                                 'optimizer': gen_optimizer.state_dict()},
                                is_best, work_dir, filename='checkpoint.pth')

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





def train(model_seg, model_ae, train_loader, epoch, criterion_seg, criterion_ae, criterion_embedding,
          optimizer_seg, optimizer_ae, logger, sublogger, trn_aelogger):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    embedding_losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    ae_ious = AverageMeter()
    ae_loss = AverageMeter()

    model_seg.train()
    model_ae.train()

    end = time.time()

    for i, (input, target,_,_) in enumerate(train_loader):

        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        #segmentation
        output_seg,bottom_seg = model_seg(input)

        # autoencoder
        if args.denoising == True:
            noisy_batch_input=utils.make_noise_input(target, args.salt_prob)
            output_ae, bottom_ae = model_ae(noisy_batch_input)

        else:
            output_ae, bottom_ae = model_ae(target)


        sig_con = True
        if args.seg_loss_function == 'bce':
            output_seg = F.sigmoid(output_seg)
            output_ae = F.sigmoid(output_ae)
            sig_con = False

            loss_seg = criterion_seg(output_seg, output_ae.detach())
            loss_ae = criterion_ae(output_ae, target)

        else:

            loss_seg = criterion_seg(output_seg, output_ae.detach())
            loss_ae = criterion_ae(F.sigmoid(output_ae), target)


        # embedding loss
        loss_embedding = utils.embedding_loss(args.embedding_loss_function, criterion_embedding,
                                        bottom_seg, bottom_ae,args.arch_ae_detach)

        total_loss = (float(args.knowledge_alpha)*(loss_seg)) + (float(args.embedding_alpha) * loss_embedding)

        print('loss_seg : ', loss_seg)
        print('loss_ae : ', loss_ae)
        print('alpha * loss_embedding : ', loss_embedding)
        print('Total_loss : ', total_loss)


        iou, dice = utils.performance(output_seg, target, dist_con=False, sig_con=sig_con)
        ae_loss.update(loss_ae.item(), input.size(0))
        embedding_losses.update(loss_embedding.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices.update(dice, input.size(0))


        ae_iou, ae_dice = utils.performance(output_ae, target, dist_con=False, sig_con=sig_con)
        losses.update(loss_ae.item(), input.size(0))
        ae_ious.update(ae_iou, input.size(0))

        optimizer_seg.zero_grad()
        optimizer_ae.zero_grad()


        # first ae backward
        if args.arch_ae_detach:
            loss_ae.backward()
        else:
            loss_ae.backward(retain_graph=True)

        optimizer_ae.step()


        # second se backward

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
            try:
                sublogger.write([epoch, i, total_loss.item(), iou, dice])
            except:
                #ipdb.set_trace()
                print('acd,asd : None')


    logger.write([epoch, losses.avg, embedding_losses.avg, ious.avg, dices.avg])
    trn_aelogger.write([epoch, ae_loss.avg, ae_ious.avg])


def validate(model,val_loader, epoch, criterion, logger):

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

            # loss-function
            output = F.sigmoid(output)

            loss = criterion(output, target)
            iou, dice = utils.performance(output, target,dist_con=False, sig_con=False)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))


            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Loss {loss.val:.4f} ({loss.avg:.4f}) '
          'IoU {ious.avg:.3f}({ious.std:.3f}) Dice {dices.avg:.3f}({dices.std:.3f}))'
        .format(loss = losses,ious=ious, dices=dices))


    logger.write([epoch, losses.avg, ious.avg, dices.avg])

    return ious.avg




if __name__ == '__main__':
    main()