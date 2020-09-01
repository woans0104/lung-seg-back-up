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
from losses import DiceLoss,tversky_loss, NLL_OHEM,JSDLoss,ClDice
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from copy import deepcopy



parser = argparse.ArgumentParser()
# arguments for dataset
parser.add_argument('--server', default='server_B')
parser.add_argument('--exp',default="test4", type=str)

parser.add_argument('--train-dataset',default='JSRT',help='JSRT_dataset,MC_dataset,SH_dataset')
parser.add_argument('--test-dataset1',default='MC_modified',help='JSRT_dataset,MC_dataset,SH_dataset')
parser.add_argument('--test-dataset2',default='SH',help='JSRT_dataset,MC_dataset,SH_dataset')


parser.add_argument('--batch-size',default=8,type=int)
parser.add_argument('--arg-mode',default=False,type=str2bool)
parser.add_argument('--arg-thres',default=0.5,type=float)
parser.add_argument('--arg-range',default='arg3',type=str)


# arguments for model
parser.add_argument('--arch-seg', default='unet', type=str)
parser.add_argument('--arch-ae', default='ae_v2', type=str)
parser.add_argument('--arch-ae-detach', default=True, type=str2bool)

parser.add_argument('--embedding-alpha', default=1, type=float)
parser.add_argument('--denoising',default=True,type=str2bool)
parser.add_argument('--salt-prob', default=0.1, type=float)

# arguments for optim & loss
parser.add_argument('--optim',default='sgd',choices=['adam','sgd'],type=str)
parser.add_argument('--eps',default=1e-08,type=float)
parser.add_argument('--weight-decay',default=5e-4,type=float)
parser.add_argument('--bce-weight', default=1, type=float)

parser.add_argument('--seg-loss-function',default='bce',type=str)
parser.add_argument('--ae-loss-function',default='bce',type=str)
parser.add_argument('--embedding-loss-function',default='mse',type=str)

parser.add_argument('--initial-lr',default=0.1,type=float)
parser.add_argument('--gamma',default=0.1,type=float)
parser.add_argument('--lr-schedule', default=[100,120], nargs='+', type=int)

parser.add_argument('--sgdMomen',default=0.9,type=float)
parser.add_argument('--clip-grad',default=False,type=str2bool)
parser.add_argument('--max-grad',default=0.5,type=float)


# arguments for dataset
parser.add_argument('--train-size',default=0.7,type=float)
parser.add_argument('--val-size',default=0,type=float)

parser.add_argument('--train-mode',default=True,type=str2bool)
parser.add_argument('--test-mode',default=True,type=str2bool)

# arguments for test mode
parser.add_argument('--inplace-test', default=1, type=int)
parser.add_argument('--file-name', default='result_all', type=str)




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

    # transform
    transform1 = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])

    # 1.train_dataset
    if args.val_size == 0:
        train_path, test_path = loader.make_dataset(args.server, args.train_dataset + '_dataset',
                                                    train_size=args.train_size)

        np.save(os.path.join(work_dir, '{}_test_path.npy'.format(args.train_dataset)), test_path)

        train_image_path = train_path[0]
        train_label_path = train_path[1]
        test_image_path = test_path[0]
        test_label_path = test_path[1]

        train_dataset = loader.CustomDataset(train_image_path, train_label_path, transform1, arg_mode=args.arg_mode,
                                             arg_thres=args.arg_thres, arg_range=args.arg_range,
                                             dataset=args.train_dataset)
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
        test_data1_path, _ = loader.make_dataset(args.server, args.test_dataset1 + '_dataset', train_size=1)
        test_data1_dataset = loader.CustomDataset(test_data1_path[0], test_data1_path[1], transform1,
                                                  dataset=args.test_dataset1)
        test_data1_loader = data.DataLoader(test_data1_dataset, batch_size=1, shuffle=True, num_workers=4)

        # 'sh'test_dataset
        test_data2_path, _ = loader.make_dataset(args.server, args.test_dataset2 + '_dataset', train_size=1)
        test_data2_dataset = loader.CustomDataset(test_data2_path[0], test_data2_path[1], transform1,
                                                  dataset=args.test_dataset2)
        test_data2_loader = data.DataLoader(test_data2_dataset, batch_size=1, shuffle=True, num_workers=0)

        test_data_list = [Train_test_loader, test_data1_loader, test_data2_loader]

        # np.save(os.path.join(work_dir, 'input_stats.npy'), train_dataset.input_stats)

        trn_logger = Logger(os.path.join(work_dir, 'train.log'))
        trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
        val_logger = Logger(os.path.join(work_dir, 'validation.log'))



    # 3.model_select
    model_seg, model_name = model_select(args.arch_seg)
    model_ae, _ = model_select(args.arch_ae)


    # 4.gpu select
    model_seg = nn.DataParallel(model_seg).cuda()
    model_ae = nn.DataParallel(model_ae).cuda()
    cudnn.benchmark = True


    # 5.optim
    if args.optim == 'adam':
        optimizer_seg = torch.optim.Adam(model_seg.parameters(), lr=args.initial_lr)
        optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=args.initial_lr)

    elif args.optim == 'sgd':
        optimizer_seg = torch.optim.SGD(model_seg.parameters(), lr=args.initial_lr, momentum=args.sgdMomen,
                                        weight_decay=args.weight_decay)
        optimizer_ae = torch.optim.SGD(model_ae.parameters(), lr=args.initial_lr, momentum=args.sgdMomen,
                                       weight_decay=args.weight_decay)


    if args.clip_grad :

        import torch.nn.utils as torch_utils
        max_grad_norm = args.max_grad

        torch_utils.clip_grad_norm_(model_seg.parameters(),
                                    max_grad_norm
                                    )
        torch_utils.clip_grad_norm_(model_ae.parameters(),
                                    max_grad_norm
                                    )

    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler_seg = optim.lr_scheduler.MultiStepLR(optimizer_seg,
                                                  milestones=lr_schedule[:-1],
                                                  gamma=args.gamma)

    lr_scheduler_ae = optim.lr_scheduler.MultiStepLR(optimizer_ae,
                                                  milestones=lr_schedule[:-1],
                                                  gamma=args.gamma)

    # 6.loss

    criterion_seg =loss_function_select(args.seg_loss_function)
    criterion_ae =loss_function_select(args.ae_loss_function)
    criterion_embedding = loss_function_select(args.embedding_loss_function)


#####################################################################################

    # train

    send_slack_message('#jm_private', '{} : starting_training'.format(args.exp))
    best_iou = 0
    try:
        if args.train_mode:
            for epoch in range(lr_schedule[-1]):


                train(model_seg =model_seg, model_ae = model_ae,train_loader=train_loader, optimizer_seg=optimizer_seg,optimizer_ae=optimizer_ae,criterion_seg=criterion_seg,criterion_ae=criterion_ae,
                      criterion_embedding=criterion_embedding,epoch= epoch, logger=trn_logger, sublogger=trn_raw_logger)

                iou = validate(model=model_seg, val_loader= val_loader,criterion = criterion_seg, epoch= epoch,logger= val_logger,work_dir=work_dir,save_fig=False,work_dir_name='{}_visualize_per_epoch'.format(args.train_dataset))
                print('validation result **************************************************************')


                lr_scheduler_seg.step()
                lr_scheduler_ae.step()

                if args.val_size == 0:
                    is_best = 1
                else:
                    is_best = iou > best_iou

                best_iou = max(iou, best_iou)
                checkpoint_filename = 'model_checkpoint_{:0>3}.pth'.format(epoch + 1)
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model_seg.state_dict(),
                                 'optimizer': optimizer_seg.state_dict()},
                                is_best,
                                work_dir,
                                filename='checkpoint.pth')



        print("train end")
    except RuntimeError as e:
        send_slack_message( '#jm_private',
                       '-----------------------------------  error train : send to message JM  & Please send a kakao talk ----------------------------------------- \n error message : {}'.format(e))

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
        main_test(model=model_seg,test_loader=test_data_list, args=args)



def model_select(network):


    # model_new

    if network == 'unet' or network == 'unet_recon':
        #my_net = Unet2D(in_shape=(1, 256, 256))
        my_net = Unet2D(in_shape=(1, 256, 256))
    elif network =='ae_v2':
        my_net= ae_lung(in_shape=(1, 256, 256))
    elif network == 'unet_tanh':
        my_net = Unet2D_tanh(in_shape=(1, 256, 256))
    elif network =='ae_tanh':
        my_net= ae_lung_tanh(in_shape=(1, 256, 256))
    elif network =='ae_shared':
        my_net= ae_lung_shared(in_shape=(1, 256, 256))
    elif network == 'unet_multy' :
        my_net = Unet2D_multipleE(in_shape=(1, 256, 256),multipleE=1)
    elif network == 'ae_multy' :
        my_net = ae_lung_multipleE(in_shape=(1, 256, 256),multipleE=1)

    else:
        raise ValueError('Not supported network.')

    model_name = str(my_net).split('(')[0]

    return my_net, model_name


def loss_function_select(loss_function):
    if loss_function == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.bce_weight])).cuda()
    elif loss_function == 'dice':
        criterion = DiceLoss().cuda()
    elif loss_function == 'mse':
        criterion = nn.MSELoss().cuda()
    elif loss_function == 'l1':
        criterion = nn.L1Loss().cuda()
    elif loss_function == 'kl' or loss_function == 'jsd':
        criterion = nn.KLDivLoss().cuda()
    elif loss_function == 'Cldice':
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.bce_weight])).cuda()
        dice = DiceLoss().cuda()
        criterion = ClDice(bce,dice,alpha=1,beta=1)
    else:
        raise ValueError('Not supported loss.')
    return criterion


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

def check_grad(model):
    ae_weight = deepcopy(model.module.decoder[2].weight)
    print('ae_weight : ', ae_weight)  # check first conv layer's first chnnel
    print('*' * 100)

    return ae_weight

def core_mode(core_mode=None,bottom_seg=None,bottom_ae=None):
    if core_mode == 'ChannelSE':
        bottom_seg = F.avg_pool2d(bottom_seg, 16)
        bottom_ae = F.avg_pool2d(bottom_ae, 16)

    elif core_mode == 'SpatialSE':
        # 1x1
        bottom_seg = torch.mean(bottom_seg, dim=1)
        bottom_ae = torch.mean(bottom_ae, dim=1)

    return bottom_seg,bottom_ae


def embedding_loss(embedding_loss,criterion_embedding, bottom_seg, bottom_ae,detach):

    if detach == True:
        bottom_ae = bottom_ae.detach()


    if embedding_loss == 'kl':
        loss_embedding = criterion_embedding(F.log_softmax(bottom_seg), F.softmax(bottom_ae))
    elif embedding_loss == 'jsd':
        loss_embedding = (0.5 * criterion_embedding(F.log_softmax(bottom_seg), F.softmax(bottom_ae))) + (
                    0.5 * criterion_embedding(F.log_softmax(bottom_seg), F.softmax(bottom_ae)))
    elif embedding_loss == 'bce':
        bottom_ae = F.sigmoid(bottom_ae).cuda()
        loss_embedding = criterion_embedding(bottom_seg, bottom_ae)  # bce
    else:
        # MSE
        loss_embedding = criterion_embedding(bottom_seg, bottom_ae)

    return loss_embedding



def train(model_seg, model_ae, train_loader, optimizer_seg, optimizer_ae, epoch, criterion_seg, criterion_ae,
                      criterion_embedding, logger, sublogger):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    embedding_losses = AverageMeter()
    recon_losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    acds = AverageMeter()
    asds = AverageMeter()

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
            noisy_batch_input=make_noise_input(target, args.salt_prob)
            output_ae, bottom_ae = model_ae(noisy_batch_input)

        else:
            output_ae, bottom_ae = model_ae(target)


        # loss-function
        if args.seg_loss_function == 'Cldice':
            loss_seg, loss_bce, loss_dice = criterion_seg(output_seg, target)
            loss_ae, _, _ = criterion_ae(output_ae, target)
        else:
            loss_seg = criterion_seg(output_seg, target)
            loss_ae = criterion_ae(output_ae, target)


        # embedding loss
        loss_embedding = embedding_loss(args.embedding_loss_function, criterion_embedding, bottom_seg, bottom_ae,args.arch_ae_detach)

        alpha = float(args.embedding_alpha)
        loss_embedding = alpha * loss_embedding



        if args.arch_seg =='unet_recon':
            recon_loss = nn.L1Loss()
            loss_recon = recon_loss(output_seg,output_ae.detach())

            recon_losses.update(loss_recon.item(), input.size(0))
            total_loss = (loss_seg) + (loss_embedding) + (loss_recon)

            print('loss_seg : ', loss_seg)
            print('loss_ae : ', loss_ae)
            print('alpha * loss_embedding : ', loss_embedding)
            print('loss_recon : ', loss_recon)
            print('Total_loss : ', total_loss)


        else:
            total_loss = (loss_seg) + (loss_embedding)

            print('loss_seg : ', loss_seg)
            print('loss_ae : ', loss_ae)
            print('alpha * loss_embedding : ', loss_embedding)
            print('Total_loss : ', total_loss)






        iou, dice,acd,asd = performance(output_seg, target)
        losses.update(total_loss.item(), input.size(0))
        embedding_losses.update(loss_embedding.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices.update(dice, input.size(0))

        if acd !=None:
            acds.update(acd, input.size(0))
        if asd !=None:
            asds.update(asd, input.size(0))

        optimizer_seg.zero_grad()
        optimizer_ae.zero_grad()


        #check_grad(model_ae)

        # first ae backward -> se backward
        if args.arch_ae_detach:
            loss_ae.backward()
        else:
            loss_ae.backward(retain_graph=True)

        optimizer_ae.step()


        #check_grad(model_ae)


        # second se backward

        total_loss.backward()
        optimizer_seg.step()


        #check_grad(model_ae)



        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'IoU {iou.val:.4f} ({iou.avg:.4f})\t'
              'Dice {dice.val:.4f} ({dice.avg:.4f})\t'
              'Acd {acd.val:.3f} ({acd.avg:.3f})\t'
              'Asd {asd.val:.3f} ({asd.avg:.3f})\t '.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses,
            iou=ious, dice=dices,acd=acds,asd=asds))

        if i % 10 == 0:
            try:
                sublogger.write([epoch, i, total_loss.item(), iou, dice,acd,asd])
            except:
                #ipdb.set_trace()
                print('acd,asd : None')

    if args.arch_seg == 'unet_recon':
        logger.write([epoch, losses.avg, embedding_losses.avg,recon_losses.avg, ious.avg, dices.avg, acds.avg, asds.avg])
    else:
        logger.write([epoch, losses.avg,embedding_losses.avg, ious.avg, dices.avg,acds.avg,asds.avg])



def validate(model,val_loader, criterion, epoch, logger,work_dir, save_fig=False, work_dir_name=False):

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
            output, _ = model(input)


            # loss-function
            if args.seg_loss_function == 'Cldice':
                loss, _, _ = criterion(output, target)

            else:
                loss = criterion(output, target)


            iou, dice,acd,asd = performance(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))
            acds.update(acd, input.size(0))
            asds.update(asd, input.size(0))


            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            if save_fig:
                if i % 10 == 0:
                    try:
                        ae_save_fig(str(epoch), ori_img, target, output, iou, dice,acd,asd, work_dir, work_dir_name,image_name[0])
                    except:
                        print('not save_fig')

            batch_time.update(time.time() - end)
            end = time.time()

    print(' * IoU {ious.avg:.3f}({ious.std:.3f}) Dice {dices.avg:.3f}({dices.std:.3f}) Acd {acds.avg:3f}({acds.std:3f}) Asd {asds.avg:3f}({asds.std:3f})'.format(
           ious=ious, dices=dices,acds=acds,asds=asds))

    logger.write([epoch, losses.avg, ious.avg, dices.avg,acds.avg,asds.avg])

    return ious.avg



def ae_save_fig(exam_id, org_input, org_target, prediction,
             iou,dice, acd,asd,result_dir,result_dir_name,slice_id):

    def _overlay_mask(img, mask, color='red'):

        # convert gray to color
        color_img = np.dstack([img, img, img])
        mask_idx = np.where(mask == 1)
        if color == 'red':
            color_img[:,:,:]= np.array([0, 0, 0])
            color_img[mask_idx[0], mask_idx[1], :] = np.array([255,0,0])
        elif color == 'blue':
            color_img[:,:,:] = np.array([0, 0, 0])
            color_img[mask_idx[0], mask_idx[1], :] = np.array([0,0,255])

        return color_img

    result_epoch_dir = os.path.join(result_dir, result_dir_name)
    if not os.path.exists(result_epoch_dir):
        os.makedirs(result_epoch_dir)
    result_exam_dir = os.path.join(result_epoch_dir, exam_id)
    if not os.path.exists(result_exam_dir):
        os.makedirs(result_exam_dir)


    assert (len(org_target) == len(prediction)), '# of results not matched.'

    prediction = prediction.squeeze().cpu().numpy()
    org_input = org_input.squeeze().cpu().numpy()
    org_target = org_target.squeeze().cpu().numpy()


    # convert prob to pred
    prediction = np.array(prediction)
    prediction = (prediction > 0.5).astype('float')



    input_slice = org_input
    target_slice = org_target
    pred_slice = prediction

    target_slice_pos_pixel =  target_slice.sum()
    target_slice_pos_pixel_rate = np.round(target_slice_pos_pixel /(512*512)*100,2)

    pred_slice_pos_pixel = pred_slice.sum()
    pred_slice_pos_pixel_rate = np.round(pred_slice_pos_pixel / (512 * 512) * 100, 2)


    fig = plt.figure(figsize=(15,5))
    ax = []
    # show original img
    ax.append(fig.add_subplot(1,3,1))
    plt.imshow(input_slice, 'gray')
    # show img with gt
    ax.append(fig.add_subplot(1,3,2))
    plt.imshow(_overlay_mask(input_slice, target_slice, color='red'))
    ax[1].set_title('GT_pos_pixel = {0}({1}%)'.format(target_slice_pos_pixel,target_slice_pos_pixel_rate))
    # show img with pred
    ax.append(fig.add_subplot(1,3,3))
    plt.imshow(_overlay_mask(input_slice, pred_slice, color='blue'))
    ax[-1].set_title('IoU = {0:.4f} \n pred_pos_pixel = {1}({2}%) \n acd ={3:.3f} asd = {4:.3f}'.format(iou, pred_slice_pos_pixel,
                                                                               pred_slice_pos_pixel_rate,acd,asd))

    # remove axis
    for i in ax:
        i.axes.get_xaxis().set_visible(False)
        i.axes.get_yaxis().set_visible(False)

    slice_id = slice_id.split('/')[-1].split('.png')[0]
    #ipdb.set_trace()
    if iou == -1:
        res_img_path = os.path.join(result_exam_dir,
                                    '{slice_id}_{iou}.png'.format(slice_id=slice_id, iou='NA'))
    else:
        res_img_path = os.path.join(result_exam_dir,
                                    '{slice_id}_{iou:.4f}.png'.format(slice_id=slice_id, iou=iou))
    plt.savefig(res_img_path, bbox_inches='tight')
    plt.close()

def make_noise_input(input,prob):
    noisy_batch_input = 0
    for j in range(input.shape[0]):
        nosiy_input = salt_and_pepper(input[j, ::],prob)

        if j == 0:
            noisy_batch_input = nosiy_input
        else:
            noisy_batch_input = torch.cat([noisy_batch_input, nosiy_input], 0)
        #plt.imshow(nosiy_input.cpu().data.numpy().reshape(256,256), 'gray')
        #plt.savefig(os.path.join(args.work_dir+'/'+args.exp,'denoising_input{}.png'.format(j)))


    return noisy_batch_input


def salt_and_pepper(img, prob):
    """salt and pepper noise for mnist"""

    c,w,h= img.shape
    rnd = np.random.rand(c*w*h)

    noisy = img.cpu().data.numpy().reshape(-1)
    noisy[rnd < prob/2] = 0.
    noisy[rnd > 1-prob/2] = 1.

    noisy = noisy.reshape(1,c,w,h)
    noisy = torch.tensor(noisy)


    return noisy



if __name__ == '__main__':
    main()