import argparse
import torch.utils.data as data
import model
import ipdb
import os
import dataloader as loader
import torchvision.transforms as transforms
import numpy as np
from medpy.metric import binary
from utils import Logger, AverageMeter, save_checkpoint ,draw_curve ,send_slack_message,str2bool
import time
import shutil
import pickle
import torch.optim as optim
from predict import main_test
from model import *
from losses import DiceLoss,tversky_loss, NLL_OHEM,ClDice
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn



parser = argparse.ArgumentParser()


# arguments for dataset
parser.add_argument('--server', default='server_B')
parser.add_argument('--exp',default="test4", type=str)


parser.add_argument('--train_mode',default=True,type=str2bool)
parser.add_argument('--test_mode',default=True,type=str2bool)

parser.add_argument('--train-dataset',default='JSRT',help='JSRT_dataset,MC_dataset,SH_dataset')
parser.add_argument('--test-dataset1',default='MC_modified',help='JSRT_dataset,MC_dataset,SH_dataset')
parser.add_argument('--test-dataset2',default='SH',help='JSRT_dataset,MC_dataset,SH_dataset')

parser.add_argument('--train-size',default=0.7,type=float)
parser.add_argument('--val-size',default=0,type=float)
parser.add_argument('--file-name', default='result_all', type=str)
parser.add_argument('--batch-size',default=8,type=int)
parser.add_argument('--arg-mode',default=False,type=str2bool)
parser.add_argument('--arg-thres',default=0.5,type=float)
parser.add_argument('--arg-range',default='arg3',type=str)


# arguments for model
parser.add_argument('--arch', default='unet', type=str)
parser.add_argument('--coordconv-no', default=[9], nargs='+', type=int)
parser.add_argument('--radious',default=False,type=str2bool)


# arguments for optim & loss
parser.add_argument('--optim',default='sgd',choices=['adam','sgd'],type=str)
parser.add_argument('--weight-decay',default=5e-4,type=float)

parser.add_argument('--loss-function',default='bce',type=str)
parser.add_argument('--bce-weight', default=1, type=float)


parser.add_argument('--initial-lr',default=0.1,type=float)
parser.add_argument('--lr-schedule', default=[100,120], nargs='+', type=int)
parser.add_argument('--clip-grad',default=False,type=str2bool)
parser.add_argument('--max-grad',default=1,type=float)


# arguments for slack
parser.add_argument('--token',type=str)


args = parser.parse_args()



def main():
    # save input stats for later use

    if args.server =='server_A':
        work_dir = os.path.join('/data1/JM/lung_segmentation', args.exp)
        print(work_dir)
    elif args.server =='server_B':
        work_dir = os.path.join('/data1/workspace/JM_gen/lung_seg', args.exp)
        print(work_dir)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # copy this file to work dir to keep training configuration
    shutil.copy(__file__, os.path.join(work_dir, 'main.py'))
    with open(os.path.join(work_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)




    # transform
    transform1 = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])

    # 1.train_dataset

    if args.val_size == 0:

        train_path, test_path = loader.make_dataset(args.server,args.train_dataset + '_dataset',train_size=args.train_size)

        np.save(os.path.join(work_dir, '{}_test_path.npy'.format(args.train_dataset)), test_path)

        train_image_path = train_path[0]
        train_label_path = train_path[1]
        test_image_path = test_path[0]
        test_label_path = test_path[1]

        train_dataset = loader.CustomDataset(train_image_path, train_label_path, transform1, arg_mode=args.arg_mode,
                                             arg_thres=args.arg_thres,arg_range=args.arg_range, dataset=args.train_dataset)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        val_dataset = loader.CustomDataset(test_image_path, test_label_path, transform1, arg_mode=False,
                                           dataset=args.train_dataset)
        val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

        # 'JSRT' test_dataset
        Train_test_dataset = loader.CustomDataset(test_image_path, test_label_path, transform1,
                                                  dataset=args.train_dataset)
        Train_test_loader = data.DataLoader(Train_test_dataset, batch_size=1, shuffle=True, num_workers=4)






    # 2.test_dataset_path


    # 'MC'test_dataset
    test_data1_path, _ = loader.make_dataset(args.server,args.test_dataset1 + '_dataset', train_size=1)
    test_data1_dataset = loader.CustomDataset(test_data1_path[0], test_data1_path[1], transform1,
                                              dataset=args.test_dataset1)
    test_data1_loader = data.DataLoader(test_data1_dataset, batch_size=1, shuffle=True, num_workers=4)

    # 'sh'test_dataset
    test_data2_path, _ = loader.make_dataset(args.server,args.test_dataset2 + '_dataset', train_size=1)
    test_data2_dataset = loader.CustomDataset(test_data2_path[0], test_data2_path[1], transform1,
                                              dataset=args.test_dataset2)
    test_data2_loader = data.DataLoader(test_data2_dataset, batch_size=1, shuffle=True, num_workers=0)

    test_data_list = [Train_test_loader, test_data1_loader, test_data2_loader]

    #np.save(os.path.join(work_dir, 'input_stats.npy'), train_dataset.input_stats)

    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))


    # 3.model_select
    my_net, model_name = model_select(args.arch)


    # 4.gpu select
    my_net = nn.DataParallel(my_net).cuda()
    cudnn.benchmark = True

    # 5.optim

    if args.optim == 'adam':
        gen_optimizer = torch.optim.Adam(my_net.parameters(), lr=args.initial_lr)
    elif args.optim == 'sgd':
        gen_optimizer = torch.optim.SGD(my_net.parameters(), lr=args.initial_lr, momentum=0.9,weight_decay=args.weight_decay)


    # gradient clipping
    if args.clip_grad :
        print('here')
        import torch.nn.utils as torch_utils
        max_grad_norm = args.max_grad

        torch_utils.clip_grad_norm_(my_net.parameters(),
                                    max_grad_norm
                                    )


    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(gen_optimizer,
                                                  milestones=lr_schedule[:-1],
                                                  gamma=0.1)

    # 6.loss
    if args.loss_function == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.bce_weight])).cuda()
    elif args.loss_function == 'dice':
        criterion = DiceLoss().cuda()
    elif args.loss_function == 'mse':
        criterion = nn.MSELoss().cuda()
    elif args.loss_function == 'Cldice':
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.bce_weight])).cuda()
        dice = DiceLoss().cuda()
        criterion = ClDice(bce,dice,alpha=1,beta=1)


#####################################################################################

    # train

    send_slack_message('#jm_private', '{} : starting_training'.format(args.exp))
    best_iou = 0
    try:
        if args.train_mode:
            for epoch in range(lr_schedule[-1]):

                train(my_net, train_loader, gen_optimizer, epoch, criterion,trn_logger, trn_raw_logger)
                iou = validate(val_loader, my_net, criterion, epoch, val_logger,work_dir=work_dir,save_fig=False,work_dir_name='jsrt_visualize_per_epoch')
                print('validation_result **************************************************************')

                lr_scheduler.step()

                if args.val_size ==0:
                    is_best = 1
                else:
                    is_best = iou > best_iou
                best_iou = max(iou, best_iou)
                checkpoint_filename = 'model_checkpoint_{:0>3}.pth'.format(epoch + 1)
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': my_net.state_dict(),
                                 'optimizer': gen_optimizer.state_dict()},
                                is_best,
                                work_dir,
                                filename='checkpoint.pth')

        print("train end")
    except RuntimeError as e:
        send_slack_message('#jm_private',
                       '-----------------------------------  error train : send to message JM  & Please send a kakao talk ----------------------------------------- \n error message : {}'.format(
                           e))
        import ipdb
        ipdb.set_trace()

    draw_curve(work_dir, trn_logger, val_logger)
    send_slack_message('#jm_private', '{} : end_training'.format(args.exp))
    #--------------------------------------------------------------------------------------------------------#
    #here is load model for last pth
    load_filename = os.path.join(work_dir, 'model_best.pth')
    checkpoint = torch.load(load_filename)
    ch_epoch = checkpoint['epoch']
    save_check_txt = os.path.join(work_dir, str(ch_epoch))
    f = open("{}_best_checkpoint.txt".format(save_check_txt), 'w')
    f.close()

    # --------------------------------------------------------------------------------------------------------#
    # validation
    if args.test_mode:
        print('Test mode ...')
        main_test(model=my_net,test_loader=test_data_list, args=args)




def train(model,train_loader,optimizer,epoch,criterion,logger, sublogger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    acds = AverageMeter()
    asds = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target,ori_input,_) in enumerate(train_loader):




        data_time.update(time.time() - end)
        input, target = input.cuda(), target.cuda()

        try:
            output,_ = model(input)
        except:
            output = model(input)


        loss = criterion(output, target)


        iou, dice,acd,asd = performance(output, target)
        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices.update(dice, input.size(0))
        acds.update(acd, input.size(0))
        asds.update(asd, input.size(0))



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
              'Dice {dice.val:.4f} ({dice.avg:.4f})\t'
              'Acd {acd.val:.4f} ({acd.avg:.4f})\t'
              'Asd {asd.val:.4f} ({asd.avg:.4f})\t'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses,
            iou=ious, dice=dices,acd=acds,asd=asds))

        if i % 10 == 0:
            try:
                sublogger.write([epoch, i, loss.item(), iou, dice, acd, asd])
            except:
                #ipdb.set_trace()
                print('acd,asd : None')

    logger.write([epoch, losses.avg, ious.avg, dices.avg,acds.avg,asds.avg])



def validate(val_loader, model, criterion, epoch, logger,work_dir,save_fig=False,work_dir_name=False):

    batch_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    acds = AverageMeter()
    asds = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target,ori_img,image_name) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            try:
                output, _ = model(input)
            except:
                output = model(input)

            loss = criterion(output, target)

            iou, dice ,acd,asd = performance(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))
            acds.update(acd, input.size(0))
            asds.update(asd, input.size(0))


            batch_time.update(time.time() - end)
            end = time.time()

    print(' * IoU {ious.avg:.3f}({ious.std:.3f}) Dice {dices.avg:.3f}({dices.std:.3f}) Acd {acds.avg:3f}({acds.std:3f}) Asd {asds.avg:3f}({asds.std:3f})'.format(
           ious=ious, dices=dices,acds=acds,asds=asds))

    logger.write([epoch, losses.avg, ious.avg, dices.avg,acds.avg,asds.avg])

    return ious.avg




def model_select(network):

    # model_new
    if network == 'unet':
        my_net = Unet2D(in_shape=(1, 256, 256))

    elif network == 'unet_coord':
        my_net = Unetcoordconv(in_shape=(1, 256, 256),coordnumber=args.coordconv_no, radius=args.radious)

    else:
        raise ValueError('Not supported network.')

    model_name = str(my_net).split('(')[0]

    return my_net, model_name



def performance(output, target):
    pos_probs = torch.sigmoid(output)
    pos_preds = (pos_probs > 0.5).float()

    pos_preds = pos_preds.cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()

    if target.sum() == 0:  # background patch
        return 0, 0

    try:
        # ACD
        acd_se = binary.assd(pos_preds, target)
        # ASD
        d_sg = np.sqrt(binary.__surface_distances(pos_preds, target, 1))
        d_gs = np.sqrt(binary.__surface_distances(target, pos_preds, 1))
        asd_se = (d_sg.sum() + d_gs.sum()) / (len(d_sg) + len(d_gs))

    except:
        #pred == 0
        acd_se =None
        asd_se = None


    # IoU
    union = ((pos_preds + target) != 0).sum()
    intersection = (pos_preds * target).sum()
    iou = intersection / union

    # dice
    dice = (2 * intersection) / (pos_preds.sum() + target.sum())

    return iou, dice,acd_se,asd_se

if __name__ == '__main__':
    main()