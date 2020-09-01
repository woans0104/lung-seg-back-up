import argparse
import torch.utils.data as data
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
from losses import DiceLoss,tversky_loss, NLL_OHEM,JSDLoss
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

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
parser.add_argument('--arch-ae', default='ae', type=str)
parser.add_argument('--load-ae-path', default='/data1/JM/lung_segmentation/autoencoder_v2/test_vae_adam_bce_0.01_v2_bce', type=str)
parser.add_argument('--embedding-alpha', default=1, type=float)
parser.add_argument('--embedding-beta', default=0.5, type=float)
parser.add_argument('--core-mode', default='N', type=str)


# arguments for optim & loss
parser.add_argument('--optim',default='sgd',choices=['adam','sgd'],type=str)
parser.add_argument('--weight-decay',default=1e-4,type=float)

parser.add_argument('--seg-loss-function',default='bce',type=str)
parser.add_argument('--ae-loss-function',default='mse',type=str)
parser.add_argument('--bce-weight', default=1, type=float)


parser.add_argument('--initial-lr',default=0.1,type=float)
parser.add_argument('--gamma',default=0.1,type=float)
parser.add_argument('--lr-schedule', default=[100,120], nargs='+', type=int)



# arguments for dataset
parser.add_argument('--train-size',default=0.7,type=float)
parser.add_argument('--val-size',default=0,type=float)

parser.add_argument('--train_mode',default=True,type=str2bool)
parser.add_argument('--test_mode',default=True,type=str2bool)

# arguments for test mode
parser.add_argument('--inplace-test', default=1, type=int)
parser.add_argument('--file-name', default='result_all', type=str)


# arguments for slack
parser.add_argument('--token',type=str)


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
    model_seg, modelName_seg = model_select(args.arch_seg)

    # load_model

    model_ae,modelName_ae = model_select(args.arch_ae)
    model_ae = nn.DataParallel(model_ae).cuda()

    checkpoint_path = os.path.join(args.load_ae_path, 'model_best.pth')
    state = torch.load(checkpoint_path)
    model_ae.load_state_dict(state['state_dict'])


    # 4.gpu select
    model_seg = nn.DataParallel(model_seg).cuda()
    cudnn.benchmark = True

    # 5.optim
    if args.optim == 'adam':
        optimizer_seg = torch.optim.Adam(model_seg.parameters(), lr=args.initial_lr)
    elif args.optim == 'sgd':
        optimizer_seg = torch.optim.SGD(model_seg.parameters(), lr=args.initial_lr, momentum=0.9,weight_decay=args.weight_decay)

    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_seg,
                                                  milestones=lr_schedule[:-1],
                                                  gamma=args.gamma)

    # 6.loss

    criterion_seg =loss_function_select(args.seg_loss_function)
    criterion_ae =loss_function_select(args.ae_loss_function)


#####################################################################################

    # train

    send_slack_message('#jm_private', '{} : starting_training'.format(args.exp))
    best_iou = 0
    try:
        if args.train_mode:
            for epoch in range(lr_schedule[-1]):

                train(model_seg, model_ae,train_loader, optimizer_seg, epoch, criterion_seg,criterion_ae,trn_logger, trn_raw_logger)
                iou = validate(val_loader, model_seg, criterion_seg, epoch, val_logger,work_dir=work_dir,save_fig=False,work_dir_name='{}_visualize_per_epoch'.format(args.train_dataset))
                print('mc_iou **************************************************************')


                lr_scheduler.step()

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
        send_slack_message('#jm_private',
                       '-----------------------------------  error train : send to message JM  & Please send a kakao talk ----------------------------------------- \n error message : {}'.format(e))

        import ipdb
        ipdb.set_trace()

    draw_curve(work_dir, trn_logger, val_logger)
    send_slack_message('#jm_private', '{} : end_training'.format(args.exp))
    #--------------------------------------------------------------------------------------------------------#
    # here is load model for last pth
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





def train(model_seg,model_ae,train_loader,optimizer_seg,epoch,criterion_seg,criterion_ae,logger, sublogger):
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
    end = time.time()

    for i, (input, target,_,_) in enumerate(train_loader):

        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()



        output_seg,bottom_seg = model_seg(input)
        loss_seg = criterion_seg(output_seg, target)


        # training model
        if args.arch_seg == 'SRM':

            output_srm, bottom_srm = model_ae(output_seg)
            _, bottom_ae = model_ae(target)
            # print(bottom_ae)
            bottom_ae = F.sigmoid(bottom_ae).cuda()
            bottom_ae = (bottom_ae > 0.5).type(torch.FloatTensor).cuda()
            loss_embedding = criterion_ae(bottom_srm, bottom_ae)  # bce

            assert bottom_srm.cpu().data.numpy().reshape(-1).min() >= 0;
            'bottom_output : {}'.format(bottom_srm.cpu().data.numpy().reshape(-1).min())
            assert bottom_ae.cpu().data.numpy().reshape(-1).min() >= 0;
            'bottom_target : {}'.format(bottom_srm.cpu().data.numpy().reshape(-1).min())

            loss_recon = DiceLoss().cuda()
            loss_recon = loss_recon(output_srm, target)

            loss_embedding = float(args.embedding_alpha) * loss_embedding
            loss_recon = float(args.embedding_beta) * loss_recon
            loss = (loss_seg) + loss_embedding + loss_recon
            recon_losses.update(loss_recon.item(), input.size(0))

            print('*' * 100)
            print('a * loss_recon : ', loss_seg)
            print('a * loss_embedding : ', loss_embedding)
            print('*'*100)


        elif args.arch_seg == 'ACNN':

            _, bottom_acnn = model_ae(output_seg)
            _, bottom_ae = model_ae(target)
            loss_embedding = ae_loss_select(args.ae_loss_function, criterion_ae, bottom_acnn, bottom_ae)
            print('loss_bottom', loss_embedding)
            loss_embedding = float(args.embedding_alpha) * loss_embedding
            loss = (loss_seg) + (loss_embedding)

        else:
            print('Not training')
            import ipdb;ipdb.set_trace()


        print('a * loss_seg : ', loss_seg)
        print('b * loss_embedding : ', loss_embedding)
        print('Toral-loss : ', loss)



        iou, dice,acd,asd = performance(output_seg, target)
        losses.update(loss.item(), input.size(0))
        embedding_losses.update(loss_embedding.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices.update(dice, input.size(0))
        if acd !=None:
            acds.update(acd, input.size(0))
        if asd !=None:
            asds.update(asd, input.size(0))

        optimizer_seg.zero_grad()
        loss.backward()
        optimizer_seg.step()

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
                sublogger.write([epoch, i, loss.item(), iou, dice,acd,asd])
            except:
                #ipdb.set_trace()
                print('acd,asd : None')


    if args.arch_seg == 'SRM':
        logger.write([epoch, losses.avg,embedding_losses.avg,recon_losses.avg, ious.avg, dices.avg,acds.avg,asds.avg])
    else:
        logger.write([epoch, losses.avg, embedding_losses.avg, ious.avg, dices.avg, acds.avg, asds.avg])


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

            output,_ = model(input)
            loss = criterion(output, target)


            iou, dice,acd,asd = performance(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))
            acds.update(acd, input.size(0))
            asds.update(asd, input.size(0))


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



def ae_loss_select(loss_name, criterion_ae, bottom_seg, bottom_ae):
    if loss_name == 'kl':
        loss_bottom = criterion_ae(F.log_softmax(bottom_seg), F.softmax(bottom_ae))
    elif loss_name == 'jsd':
        loss_bottom = (0.5 * criterion_ae(F.log_softmax(bottom_seg), F.softmax(bottom_ae))) + (
                0.5 * criterion_ae(F.log_softmax(bottom_ae), F.softmax(bottom_seg)))
    else:
        # MSE
        loss_bottom = criterion_ae(bottom_seg, bottom_ae)

    return loss_bottom


def core_mode(core_mode=args.core_mode,bottom_seg=None,bottom_ae=None):
    if core_mode == 'ChannelSE':
        bottom_seg = F.avg_pool2d(bottom_seg, 16)
        bottom_ae = F.avg_pool2d(bottom_ae, 16)

    elif core_mode == 'SpatialSE':
        # 1x1
        bottom_seg = torch.mean(bottom_seg, dim=1)
        bottom_ae = torch.mean(bottom_ae, dim=1)

    return bottom_seg,bottom_ae


def model_select(network):

    # model_new
    if network == 'unet' or network == 'ACNN' or network == 'SRM':
        my_net = Unet2D(in_shape=(1, 256, 256))

    elif network == 'ae_v2':
        my_net = ae_lung(in_shape=(1, 256, 256))

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
if __name__ == '__main__':
    main()