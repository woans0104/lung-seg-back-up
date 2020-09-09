import os
import glob
import argparse
import json
import ipdb

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from medpy.metric import binary

from model import *
from utils import *
import dataloader as loader


def main_test(model=None, args=None, test_loader=None):
    if args.server == 'server_A':
        work_dir = os.path.join('/data1/JM/lung-seg-back-up', args.exp)
        print(work_dir)
    elif args.server == 'server_B':
        work_dir = os.path.join('/data1/workspace/JM_gen/lung-seg-back-up', args.exp)
        print(work_dir)
    elif args.server == 'server_D':
        work_dir = os.path.join('/daintlab/home/shwang/workspace/lung-seg-back-up', args.exp)
        print(work_dir)

    file_name = args.file_name

    result_dir = os.path.join(work_dir, file_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # load model and input stats

    #if model is None:
    #    model = select_model(args.arch)
    #    model = nn.DataParallel(model).cuda()
    model_encoder = Unet2D_encoder(in_shape=(1,256,256))
    model_decoder = Unet2D_decoder(in_shape=(1,256,256))
    model_encoder = nn.DataParallel(model_encoder).cuda()
    model_decoder = nn.DataParallel(model_decoder).cuda()

    checkpoint_path = os.path.join(work_dir, 'model_best.pth')
    state = torch.load(checkpoint_path)
    model_encoder.load_state_dict(state['state_dict'][0])
    model_decoder.load_state_dict(state['state_dict'][1])
    cudnn.benchmark = True

    source_dataset, target_dataset1, target_dataset2 = loader.dataset_condition(args.source_dataset)
    test_data_name_li = [source_dataset, target_dataset1, target_dataset2]

    collated_performance = {}
    for i in range(len(test_data_name_li)):

        if test_loader is None:
            prediction_li, org_input_li, org_target_li, img_name_li = predict(server=args.server, work_dir=work_dir,
                                                                              model=[model_encoder, model_decoder],
                                                                              exam_root=test_data_name_li[i], args=args)

        else:
            prediction_li, org_input_li, org_target_li, img_name_li = predict(server=args.server, work_dir=work_dir,
                                                                              model=[model_encoder, model_decoder], exam_root=None,
                                                                              tst_loader=test_loader[i], args=args)

        # measure performance
        performance = performance_by_slice(prediction_li, org_target_li, img_name_li)

        result_dir_sep = os.path.join(result_dir, test_data_name_li[i])
        if not os.path.exists(result_dir_sep):
            os.makedirs(result_dir_sep)

        #save_fig(org_input_li, org_target_li, prediction_li, performance, result_dir_sep)
        collated_performance[test_data_name_li[i]] = performance

    # save_result
    import pandas as pd

    df = pd.DataFrame(columns=['IOU', 'DICE', 'ACD', 'ASD', 'ACC'])
    for h in collated_performance.keys():
        overal_performance = compute_overall_performance(collated_performance[h])

        df.loc[h] = [overal_performance['segmentation_performance'][0],
                     overal_performance['segmentation_performance'][1],
                     overal_performance['distance_performance[acd,asd]'][0],
                     overal_performance['distance_performance[acd,asd]'][1],
                     overal_performance['slice_level_accuracy']]

        with open(os.path.join(result_dir, '{}_performance.json'.format(h)), 'w') as f:
            json.dump(overal_performance, f)

    df.to_csv(os.path.join(result_dir, 'lungseg_performance.csv'), mode='w')


def predict(server, work_dir, model, exam_root, tst_loader=None, args=None):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    if tst_loader == None:

        try:
            npy_file = sorted(glob.glob(work_dir + '/*.npy'))
        except:
            ipdb.set_trace()

        import ipdb; ipdb.set_trace()

        if exam_root == npy_file[0].lower().split('/')[-1].split('_')[0]:
            tst_data_path = np.load(npy_file[0]).tolist()
            tst_img_data_path, tst_label_data_path = tst_data_path

            tst_dataset = loader.Lung_Dataset(tst_img_data_path, tst_label_data_path, transform, dataset=exam_root)
            tst_loader = data.DataLoader(tst_dataset, batch_size=1, shuffle=False, num_workers=0)
        else:
            tst_loader, _ = loader.get_loader(server=server, dataset=exam_root, train_size=1, batch_size=1,
                                              aug_mode=False, aug_range=None)

    print('exam_root', exam_root)
    print(len(tst_loader))

    prob_img_li = []
    input_img_li = []
    target_img_li = []
    image_name_li = []

    model_encoder, model_decoder = model
    model_encoder.eval()
    model_decoder.eval()

    #model.eval()
    with torch.no_grad():
        for i, (input, target, img_ori, image_name) in enumerate(tst_loader):

            input = input.cuda()

            #try:
            center, intermediate = model_encoder(input)
            output = model_decoder(center, intermediate)
            #except:
                #output, _ = model(input)
            #    pass

            # convert to prob

            pos_probs = torch.sigmoid(output[0])
            pos_probs = pos_probs.squeeze().cpu().numpy()
            img_ori = img_ori.squeeze().cpu().numpy()
            target = target.squeeze().cpu().numpy()

            prob_img_li.append(pos_probs)
            input_img_li.append(img_ori)
            target_img_li.append(target)

            image_name = image_name[0].split('/')[-1].split('.png')[0]
            image_name_li.append(image_name)

        print('end---------')
        return prob_img_li, input_img_li, target_img_li, image_name_li


def performance_by_slice(output_list, target_list, img_name_list):
    assert len(output_list) == len(target_list), 'not same list lenths'

    performance = {}
    for i in range(len(output_list)):
        preds = output_list[i]
        slice_pred = (preds > 0.5).astype('float')
        slice_target = target_list[i]

        # slice-level classification performance
        tp = fp = tn = fn = 0
        is_gt_positive = slice_target.max()
        is_pred_positive = slice_pred.max()
        if is_gt_positive:
            if is_pred_positive:
                tp = 1
            else:
                fn = 1
        else:
            if is_pred_positive:
                fp = 1
            else:
                tn = 1

        # slice-level segmentation performance
        iou = dice = -1
        if is_gt_positive:
            union = ((slice_pred + slice_target) != 0).sum()
            intersection = (slice_pred * slice_target).sum()

            iou = intersection / union
            dice = (2 * intersection) / (slice_pred.sum() + slice_target.sum())

            try:
                # ACD
                acd_se = binary.assd(slice_pred, slice_target)

                # ASD
                d_sg = np.sqrt(binary.__surface_distances(slice_pred, slice_target, 1))
                d_gs = np.sqrt(binary.__surface_distances(slice_target, slice_pred, 1))
                asd_se = (d_sg.sum() + d_gs.sum()) / (len(d_sg) + len(d_gs))

            except:
                # pred == 0
                acd_se = None
                asd_se = None

        # TODO: not need to store gt and pred
        performance[str(i)] = {'cls': [tp, fp, tn, fn],
                               'seg': [iou, dice],
                               'gt': slice_target,
                               'pred': slice_pred,
                               'img': img_name_list[i],
                               'acd_se': acd_se,
                               'asd_se': asd_se,
                               }

    return performance


def compute_overall_performance(collated_performance):
    confusion_matrix = np.zeros((4,))
    iou_sum = dice_sum = n_valid_slices = acd_sum = asd_sum = distanse_count = 0
    for res_slice in collated_performance.values():
        confusion_matrix += np.array(res_slice['cls'])
        if res_slice['gt'].sum() != 0:  # consider only annotated slices
            iou_sum += res_slice['seg'][0]
            dice_sum += res_slice['seg'][1]

            n_valid_slices += 1

            if res_slice['acd_se'] == None or res_slice['asd_se'] == None:
                continue
            acd_sum += res_slice['acd_se']
            asd_sum += res_slice['asd_se']
            distanse_count += 1

    iou_mean = np.round(iou_sum / n_valid_slices, 3)
    dice_mean = np.round(dice_sum / n_valid_slices, 3)
    acd_se_mean = np.round(acd_sum / distanse_count, 3)
    asd_se_mean = np.round(asd_sum / distanse_count, 3)

    return {'confusion_matrix': list(confusion_matrix),
            'slice_level_accuracy': (confusion_matrix[0] + confusion_matrix[2]) / confusion_matrix.sum(),
            'segmentation_performance': [iou_mean, dice_mean],
            'distance_performance[acd,asd]': [acd_se_mean, asd_se_mean]}


def save_fig(org_input, org_target, prediction,
             slice_level_performance, result_dir):
    def _overlay_mask(img, mask, color='red'):

        # convert gray to color
        color_img = np.dstack([img, img, img])
        mask_idx = np.where(mask == 1)
        if color == 'red':
            color_img[mask_idx[0], mask_idx[1], :] = np.array([255, 0, 0])
        elif color == 'blue':
            color_img[mask_idx[0], mask_idx[1], :] = np.array([0, 0, 255])

        return color_img

    assert (len(org_target) == len(prediction) \
            == len(slice_level_performance)), '# of results not matched.'

    # convert prob to pred

    prediction = np.array(prediction)
    prediction = (prediction > 0.5).astype('float')

    for slice_id in slice_level_performance:

        iou, dice = slice_level_performance[slice_id]['seg']
        acd = slice_level_performance[slice_id]['acd_se']
        asd = slice_level_performance[slice_id]['asd_se']
        img_name = slice_level_performance[slice_id]['img']
        input_slice = org_input[int(slice_id)]
        target_slice = org_target[int(slice_id)]
        pred_slice = prediction[int(slice_id)]

        target_slice_pos_pixel = target_slice.sum()
        target_slice_pos_pixel_rate = np.round(target_slice_pos_pixel / (256 * 256) * 100, 2)

        pred_slice_pos_pixel = pred_slice.sum()
        pred_slice_pos_pixel_rate = np.round(pred_slice_pos_pixel / (256 * 256) * 100, 2)

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
        try:
            ax[-1].set_title('IoU = {0:.4f} \n pred_pos_pixel = {1}({2}%) '
                             '\n acd ={3:.3f} asd = {4:.3f}'
                             .format(iou, pred_slice_pos_pixel, pred_slice_pos_pixel_rate, acd, asd))
        except:
            ax[-1].set_title('IoU = {0:.4f} \n pred_pos_pixel = {1}({2}%) '
                             '\n acd =None asd = None'.format(iou, pred_slice_pos_pixel, pred_slice_pos_pixel_rate))

        # remove axis
        for i in ax:
            i.axes.get_xaxis().set_visible(False)
            i.axes.get_yaxis().set_visible(False)

        if iou == -1:
            res_img_path = os.path.join(result_dir,
                                        'FILE{slice_id}_{iou}.png'.format(slice_id=img_name, iou='NA'))
        else:
            res_img_path = os.path.join(result_dir,
                                        'FILE{slice_id}_{iou:.4f}.png'.format(slice_id=img_name, iou=iou))
        plt.savefig(res_img_path, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='server_D')
    parser.add_argument('--exp', type=str)
    parser.add_argument('--file-name', default='result_all_acd', type=str)

    parser.add_argument('--source-dataset', default='JSRT', help='JSRT_dataset,MC_dataset,SH_dataset')

    parser.add_argument('--arch', default='unet', type=str)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--style', default=0, type=int)

    args = parser.parse_args()

    main_test(args=args)
    # test24_diceloss
