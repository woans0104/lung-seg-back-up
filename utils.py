
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import shutil
from collections import Iterable
import argparse

from model import *
from losses import DiceLoss,ClDice

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0 # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val!=None: # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val**2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2/self.count - self.avg**2)
        else:
            pass

class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.',v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log


def calc_stats(data_root):

    data_ids = os.listdir(os.path.join(data_root, 'images'))

    mean_meter = AverageMeter()
    std_meter = AverageMeter()

    for data_id in data_ids:
        image_dir = os.path.join(data_root, 'images', data_id)
        image_3d = load_exam(image_dir, ftype='png')
        pixel_mean = image_3d.mean()
        pixel_std = image_3d.std()

        mean_meter.update(pixel_mean, image_3d.size)
        std_meter.update(pixel_std, image_3d.size)

    total_mean = mean_meter.avg
    total_std = np.sqrt(std_meter.sum_2/std_meter.count)

    return {'mean': total_mean, 'std': total_std}



def performance(output, target, dist_con, sig_con=True):

    if sig_con:
        pos_probs = torch.sigmoid(output)
    else:
        pos_probs = output

    pos_preds = (pos_probs > 0.5).float()

    pos_preds = pos_preds.cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()

    if target.sum() == 0:  # background patch
        return 0, 0


    # IoU
    union = ((pos_preds + target) != 0).sum()
    intersection = (pos_preds * target).sum()
    iou = intersection / union

    # dice
    dice = (2 * intersection) / (pos_preds.sum() + target.sum())

    if dist_con == False:
        return iou, dice

    else:
        #mode == 'test':

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

        return iou, dice, acd_se, asd_se



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


#
# def send_slack_message(channel, messge,
#                        token='str'):
#     token = token
#     slack = Slacker(token)
#     slack.chat.post_message(channel, messge)



def save_checkpoint(state, is_best, work_dir, filename='checkpoint.pth'):
    checkpoint_path = os.path.join(work_dir, filename)
    if is_best:
        torch.save(state, checkpoint_path)
        shutil.copyfile(checkpoint_path,
                        os.path.join(work_dir, 'model_best.pth'))


def check_best_pth(work_dir):
    load_filename = os.path.join(work_dir, 'model_best.pth')
    checkpoint = torch.load(load_filename)
    ch_epoch = checkpoint['epoch']
    save_check_txt = os.path.join(work_dir, str(ch_epoch))
    f = open("{}_best_checkpoint.txt".format(save_check_txt), 'w')
    f.close()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def embedding_regularize(vector, mode):
    if mode == 'l1':
        regularize_vector = torch.norm(vector, p=1)
    elif mode == 'l2':
        regularize_vector = torch.norm(vector, p=2)

    return regularize_vector



def gramMatrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    features = input.view(a,b ,c*d)
    features_t = features.transpose(1, 2)
    G = features.bmm(features_t)  # compute the gram product

    return G.div(b * c * d)



def select_model(network,style='style1'):

    # model_new

    if network == 'unet' or network == 'unet_recon' or network == 'ACNN' or network == 'SRM':
        my_net = Unet2D(in_shape=(1, 256, 256))
    elif network =='ae_v2':
        my_net= ae_lung(in_shape=(1, 256, 256))

    elif network == 'unet_style':
        my_net = Unet2D_style(in_shape=(1, 256, 256), style=style)
    elif network == 'ae_v2_style':
        my_net = ae_lung_style(in_shape=(1, 256, 256), style=style)

    else:
        raise ValueError('Not supported network.')



    return my_net


# def select_loss(loss_function):
#     if loss_function == 'bce':
#         criterion = nn.BCELoss()
#     elif loss_function == 'bce_logit':
#         criterion = nn.BCEWithLogitsLoss()
#     elif loss_function == 'dice':
#         criterion = DiceLoss()
#     elif loss_function == 'mse':
#         criterion = nn.MSELoss()
#     elif loss_function == 'l1':
#         criterion = nn.L1Loss()
#     elif loss_function == 'kl' or loss_function == 'jsd':
#         criterion = nn.KLDivLoss()
#     elif loss_function == 'Cldice':
#         bce = nn.BCELoss()
#         dice = DiceLoss()
#         criterion = ClDice(bce,dice,alpha=1,beta=1)
#     else:
#         raise ValueError('Not supported loss.')
#     return criterion



def embedding_loss(embedding_loss,criterion_embedding, bottom_seg, bottom_ae, detach):

    if detach == True:
        bottom_ae = bottom_ae.detach()

    if embedding_loss == 'kl':
        #loss_embedding = criterion_embedding(F.log_softmax(bottom_seg), F.softmax(bottom_ae))
        loss_embedding = criterion_embedding(F.log_softmax(bottom_seg.flatten(start_dim=2), dim=2).view_as(bottom_seg),
                                             F.softmax(bottom_ae.flatten(start_dim=2), dim=2).view_as(bottom_ae))
    elif embedding_loss == 'jsd':
        loss_embedding = (0.5 * criterion_embedding(F.log_softmax(bottom_seg), F.softmax(bottom_ae))) \
                         + ( 0.5 * criterion_embedding(F.log_softmax(bottom_seg), F.softmax(bottom_ae)))

    elif embedding_loss == 'bce':
        bottom_seg = F.sigmoid(bottom_seg).cuda()
        bottom_ae = F.sigmoid(bottom_ae).cuda()
        loss_embedding = criterion_embedding(bottom_seg, bottom_ae)  # bce
    elif embedding_loss == 'cos':
        cos_sim = criterion_embedding(bottom_seg, bottom_ae).mean()
        loss_embedding = 1 - cos_sim
    else:
        # MSE
        loss_embedding = criterion_embedding(bottom_seg, bottom_ae)

    return loss_embedding


def save_fig(exam_id, org_input, org_target, prediction, iou, result_dir, work_dir_name, slice_id):
    def _overlay_mask(img, mask, color='red'):

        # convert gray to color
        color_img = np.dstack([img, img, img])
        mask_idx = np.where(mask == 1)
        if color == 'red':
            color_img[:, :, :] = np.array([0, 0, 0])
            color_img[mask_idx[0], mask_idx[1], :] = np.array([255, 0, 0])
        elif color == 'blue':
            color_img[:, :, :] = np.array([0, 0, 0])
            color_img[mask_idx[0], mask_idx[1], :] = np.array([0, 0, 255])

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

    target_slice_pos_pixel = target_slice.sum()
    target_slice_pos_pixel_rate = np.round(target_slice_pos_pixel / (512 * 512) * 100, 2)

    pred_slice_pos_pixel = pred_slice.sum()
    pred_slice_pos_pixel_rate = np.round(pred_slice_pos_pixel / (512 * 512) * 100, 2)

    fig = plt.figure(figsize=(15, 5))
    ax = []
    # show original img
    ax.append(fig.add_subplot(1, 3, 1))
    plt.imshow(input_slice, 'gray')
    # show img with gt
    ax.append(fig.add_subplot(1, 3, 2))
    plt.imshow(_overlay_mask(input_slice, target_slice, color='red'))
    ax[1].set_title('GT_pos_pixel = {0}({1}%)'.format(target_slice_pos_pixel, target_slice_pos_pixel_rate))
    # show img with pred
    ax.append(fig.add_subplot(1, 3, 3))
    plt.imshow(_overlay_mask(input_slice, pred_slice, color='blue'))
    ax[-1].set_title('IoU = {0:.4f} \n pred_pos_pixel = {1}({2}%)'.format(iou, pred_slice_pos_pixel,
                                                                          pred_slice_pos_pixel_rate))

    # remove axis
    for i in ax:
        i.axes.get_xaxis().set_visible(False)
        i.axes.get_yaxis().set_visible(False)

    slice_id = slice_id.split('/')[-1].split('.png')[0]
    # ipdb.set_trace()
    if iou == -1:
        res_img_path = os.path.join(result_exam_dir,
                                    '{slice_id}_{iou}.png'.format(slice_id=slice_id, iou='NA'))
    else:
        res_img_path = os.path.join(result_exam_dir,
                                    '{slice_id}_{iou:.4f}.png'.format(slice_id=slice_id, iou=iou))
    plt.savefig(res_img_path, bbox_inches='tight')
    plt.close()


def draw_curve(work_dir, logger1, logger2, labelname=None):
    logger1 = logger1.read()
    logger2 = logger2.read()

    if len(logger1[0]) == 3:
        epoch, trn_loss1, iou1 = zip(*logger1)
        epoch, trn_loss2, iou2 = zip(*logger2)

    elif len(logger1[0]) == 5:
        epoch, trn_loss1, embedding_loss1, iou1, dice1 = zip(*logger1)
        epoch, trn_loss2, iou2, dice2 = zip(*logger2)

    elif len(logger1[0]) == 6:
        epoch, trn_loss1, embedding_loss1, recon_losses, iou1, dice1 = zip(*logger1)
        epoch, trn_loss2, iou2, dice2, acd2, asd2 = zip(*logger2)


    else:
        epoch, trn_loss1, iou1, dice1 = zip(*logger1)
        epoch, trn_loss2, iou2, dice2 = zip(*logger2)

    plt.figure(1)
    plt.plot(epoch, trn_loss1, '-b', label='train_total_loss')
    plt.plot(epoch, trn_loss2, '-r', label='val_loss')
    if len(logger1[0]) == 5:
        plt.plot(epoch, embedding_loss1, '-g', label='train_embedding_loss')
    elif len(logger1[0]) == 6:
        plt.plot(epoch, embedding_loss1, '-g', label='train_embedding_loss')
        plt.plot(epoch, recon_losses, '-y', label='train_recon_loss')

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('compare_loss')
    plt.savefig(os.path.join(work_dir, 'trn_loss.png'))

    plt.figure(2)
    plt.plot(epoch, iou1, '-b', label='train-{}'.format(labelname))
    plt.plot(epoch, iou2, '-r', label='val-{}'.format(labelname))

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('compare_iou')
    plt.savefig(os.path.join(work_dir, 'compare_val_perf.png'))

    plt.close()

