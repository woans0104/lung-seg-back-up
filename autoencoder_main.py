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
from autoencoder_pred import main_test
from model import *
from losses import DiceLoss,tversky_loss, NLL_OHEM,ClDice
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torchvision


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
parser.add_argument('--inplace-test', default=1, type=int)
parser.add_argument('--file-name', default='result_all', type=str)
parser.add_argument('--batch-size',default=8,type=int)
parser.add_argument('--arg-mode',default=False,type=str2bool)
parser.add_argument('--arg-thres',default=0.5,type=float)
parser.add_argument('--arg-range',default='arg3',type=str)


# arguments for model
parser.add_argument('--arch', default='unet', type=str)
parser.add_argument('--sparse',default=False,type=str2bool)
parser.add_argument('--denoising',default=False,type=str2bool)
parser.add_argument('--salt-prob', default=0.1, type=float)
parser.add_argument('--sparse-beta', default=1, type=float)

# arguments for optim & loss
parser.add_argument('--optim',default='sgd',choices=['adam','sgd'],type=str)
parser.add_argument('--eps',default=1e-08,type=float)
parser.add_argument('--weight-decay',default=1e-4,type=float)

parser.add_argument('--loss-function',default='bce',type=str)
parser.add_argument('--bce-weight', default=1, type=float)



parser.add_argument('--initial-lr',default=0.1,type=float)
parser.add_argument('--lr-schedule', default=[100,120], nargs='+', type=int)
parser.add_argument('--gamma',default=0.1,type=float)


# arguments for slack
parser.add_argument('--token',type=str)


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
    my_net, model_name = model_select(args.arch)

    # 4.gpu select
    my_net = nn.DataParallel(my_net).cuda()
    cudnn.benchmark = True
    # 5.optim
    gen_optimizer = optim_select(args.optim, my_net, args.initial_lr, args.eps)

    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(gen_optimizer,
                                                  milestones=lr_schedule[:-1],
                                                  gamma=args.gamma)

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
        criterion = ClDice(bce, dice, alpha=1, beta=1)
    else:
        #l1
        criterion = nn.L1Loss().cuda()


#####################################################################################

    # train

    send_slack_message('#jm_private', '{} : starting_training'.format(args.exp))
    best_iou = 0
    try:
        if args.train_mode:
            for epoch in range(lr_schedule[-1]):

                train(my_net, train_loader, gen_optimizer, epoch, criterion,trn_logger, trn_raw_logger)
                iou = validate(val_loader, my_net, criterion, epoch, val_logger,work_dir=work_dir,save_fig=True,work_dir_name='{}_visualize_per_epoch'.format(args.train_dataset))
                print('mc_iou **************************************************************')

                lr_scheduler.step()

                if args.val_size == 0:
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
        main_test(model=my_net,test_loader=test_data_list, args=args)


def train(model,train_loader,optimizer,epoch,criterion,logger, sublogger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    model.train()
    end = time.time()
    core_mean=0
    for i, (input, target,_,_) in enumerate(train_loader):

        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        if args.denoising == True:
            noisy_batch_input = make_noise_input(target)

            output, _ = model(noisy_batch_input)
            loss = criterion(output, target)


        else:
            #cae

            output, core = model(target)
            loss = criterion(output, target)


        iou, dice = performance(output, target)
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




def validate(val_loader, model, criterion, epoch, logger,work_dir,save_fig=False,work_dir_name=False):

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

            iou, dice = performance(output, target)
            print(iou)
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))

            if save_fig:
                if i % 10 == 0:
                    ae_save_fig(str(epoch), ori_img, target, output, iou, dice, work_dir,work_dir_name, image_name[0])

            batch_time.update(time.time() - end)
            end = time.time()

    print(' * IoU {ious.avg:.3f}({ious.std:.3f}) Dice {dices.avg:.3f}({dices.std:.3f})'.format(
           ious=ious, dices=dices))

    logger.write([epoch, losses.avg, ious.avg, dices.avg])


    return ious.avg


def model_select(network):


    if network == 'ae_v2':
        my_net = ae_lung(in_shape=(1, 256, 256))
    elif network == 'ae_tanh':
        my_net = ae_lung_tanh(in_shape=(1, 256, 256))
    elif network == 'ae_shared':
        my_net = ae_lung_shared(in_shape=(1, 256, 256))
    else:
        raise ValueError('Not supported network.')

    model_name = str(my_net).split('(')[0]

    return my_net, model_name





def optim_select(optim_name, my_net, lr, eps):
    if optim_name == 'adam':
        gen_optimizer = torch.optim.Adam(my_net.parameters(), lr=lr, eps=eps,weight_decay=args.weight_decay)
    elif optim_name == 'sgd':
        gen_optimizer = torch.optim.SGD(my_net.parameters(), lr=lr, momentum=0.9,weight_decay=args.weight_decay)

    return gen_optimizer




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

    return iou, dice


def make_noise_input(input):
    noisy_batch_input = 0
    for j in range(input.shape[0]):
        nosiy_input = salt_and_pepper(input[j, ::], args.salt_prob)
        if j == 0:
            noisy_batch_input = nosiy_input
        else:
            noisy_batch_input = torch.cat([noisy_batch_input, nosiy_input], 0)
        # plt.imshow(nosiy_input.cpu().data.numpy().reshape(256,256), 'gray')
        # plt.savefig(os.path.join(args.work_dir+'/'+args.exp,'denoising_input{}.png'.format(i)))


    #ipdb.set_trace()
    return noisy_batch_input


def cvae_mode(model,input,target,cvae_mode):

    if cvae_mode == 'target_coordconv':
        print('check')
        print('check', target.shape)
        output, mu, logvar, core = model(target, target)
    elif cvae_mode == 'input_coordconv':
        output, mu, logvar, core = model(target, input)
    elif cvae_mode == 'input_plus_target_coordconv' or cvae_mode == 'input_plus_target':
        input = torch.cat([input,target],1)

        output, mu, logvar, core = model(target, input)
    else:
        output, mu, logvar, core = model(target, input)


    return output, mu, logvar

def save_layer_fig(model,exam_id,result_dir,work_dir_name):

    result_epoch_dir = os.path.join(result_dir, work_dir_name)
    if not os.path.exists(result_epoch_dir):
        os.makedirs(result_epoch_dir)
    result_exam_dir = os.path.join(result_epoch_dir, exam_id)
    if not os.path.exists(result_exam_dir):
        os.makedirs(result_exam_dir)

    res_img_path = os.path.join(result_exam_dir,
                                'visualize_encoderlayer.png')

    data = np.transpose(torchvision.utils.make_grid(model.module.encoder[0].weight.cpu().data).numpy(),(1,2,0))
    d_min,d_max = data.min(),data.max()
    d_nomalize = (data-d_min) / (d_max-d_min)

    plt.figure(figsize=(80,100))
    plt.imshow(d_nomalize)
    plt.axis('off')
    plt.savefig(res_img_path, bbox_inches='tight')
    plt.close()

def ae_save_fig(exam_id, org_input, org_target, prediction,
             iou,dice, result_dir,work_dir_name,slice_id):

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

    result_epoch_dir = os.path.join(result_dir, work_dir_name)
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
    ax[-1].set_title('IoU = {0:.4f} \n pred_pos_pixel = {1}({2}%)'.format(iou, pred_slice_pos_pixel,
                                                                               pred_slice_pos_pixel_rate))

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




def sparse_loss(autoencoder, images):
    loss = 0
    values = images
    for i in range(3):
        fc_layer = list(autoencoder.encoder.children())[2 * i]
        relu = list(autoencoder.encoder.children())[2 * i + 1]
        values = fc_layer(values)
        loss += kl_divergence(0.03, values)
    for i in range(2):
        fc_layer = list(autoencoder.decoder.children())[2 * i]
        relu = list(autoencoder.decoder.children())[2 * i + 1]
        values = fc_layer(values)
        loss += kl_divergence(0.03, values)
    return loss


def kl_divergence(p, p_hat):
    funcs = nn.Sigmoid()

    p_hat = p_hat.view(-1)
    p_hat = p_hat.unsqueeze(0)
    p_hat = torch.mean(funcs(p_hat), 1)
    p_tensor = torch.Tensor([p] * len(p_hat)).cuda()
    return torch.sum(p_tensor * torch.log(p_tensor) - p_tensor * torch.log(p_hat) + (1 - p_tensor) * torch.log(1 - p_tensor) - (1 - p_tensor) * torch.log(1 - p_hat))

def salt_and_pepper(img, prob):

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