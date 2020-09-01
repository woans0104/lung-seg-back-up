import os
import argparse
import json
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model
from model import *
from utils import Logger,str2bool
import ipdb
import dataloader as loader
import torchvision.transforms as transforms
from medpy.metric import binary


def main_test(model=None, args=None,test_loader=None,val_mode=False):

    if args.server == 'server_A':
        work_dir = os.path.join('/data1/JM/lung_segmentation', args.exp)
        print(work_dir)
    elif args.server == 'server_B':
        work_dir = os.path.join('/data1/workspace/JM_gen/lung_seg', args.exp)
        print(work_dir)


    file_name = args.file_name
    result_dir = os.path.join(work_dir, file_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # load model and input stats
    # Note: here, the model should be given manually
    # TODO: try to import model configuration later

    if model is None:
        model,model_name = load_model(args.arch)
        model = nn.DataParallel(model).cuda()
        print('+++++++',model_name,"+++++++++")

    checkpoint_path = os.path.join(work_dir, 'model_best.pth')
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    cudnn.benchmark = True


    collated_performance = {}
    result_dir_list = []

    if test_loader is None:
        # list exam ids
        test_data_name_list = []
        for i in range(len(args.test_root)):
            test_data_name = args.test_root[i].split('/')[-1]
            prediction_list, org_input_list, org_target_list,img_name_list = predict(args.server,work_dir,model, test_data_name, args=args)

            # measure performance
            performance = performance_by_slice(prediction_list, org_target_list,img_name_list)

            result_dir_sep =os.path.join(result_dir,test_data_name)
            if not os.path.exists(result_dir_sep):
                os.makedirs(result_dir_sep)
            result_dir_list.extend(result_dir_sep)
            test_data_name_list.append(test_data_name)

            save_fig(org_input_list, org_target_list, prediction_list, performance, result_dir_sep)
            collated_performance[test_data_name] = performance

    else:
        try:
            test_data_name_list = [args.train_dataset, args.test_dataset1, args.test_dataset2]
        except:
            import ipdb;ipdb.set_trace()
            print('Not dataset_name')

        for i in range(len(test_loader)):

            prediction_list, org_input_list, org_target_list,img_name_list = predict(args.server,work_dir,model, exam_root=None,tst_loader=test_loader[i], args=args)

            # measure performance
            performance = performance_by_slice(prediction_list, org_target_list,img_name_list)

            result_dir_sep = os.path.join(result_dir, test_data_name_list[i])
            if not os.path.exists(result_dir_sep):
                os.makedirs(result_dir_sep)
            result_dir_list.extend(result_dir_sep)

            save_fig(org_input_list, org_target_list, prediction_list, performance, result_dir_sep)
            collated_performance[test_data_name_list[i]] = performance



    # save_result
    import pandas as pd

    df = pd.DataFrame(columns = ['IOU', 'DICE', 'ACD', 'ASD','ACC'])
    for h in collated_performance.keys():
        overal_performance = compute_overall_performance(collated_performance[h])

        df.loc[h] = [ overal_performance['segmentation_performance'][0], overal_performance['segmentation_performance'][1],
                      overal_performance['distance_performance[acd,asd]'][0],overal_performance['distance_performance[acd,asd]'][1],overal_performance['slice_level_accuracy']]

        with open(os.path.join(result_dir, '{}_performance.json'.format(h)), 'w') as f:
            json.dump(overal_performance, f)

    df.to_csv(os.path.join(result_dir, 'lungseg_performance.csv'),mode='w')




def predict(server,work_dir,model, exam_root,tst_loader=None, args=None):

    transform1 = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5],[0.5])])
    if tst_loader == None:

        try:
            npy_file = sorted(glob.glob(work_dir + '/*.npy'))
        except:
            ipdb.set_trace()

        if exam_root.lower().split('_')[0] == npy_file[0].lower().split('/')[-1].split('_')[0]:
            test_data_path = np.load(npy_file[0]).tolist()
            test_dataset = loader.CustomDataset(test_data_path[0], test_data_path[1], transform1,dataset=exam_root.lower().split('_')[0])
            tst_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
        else:
            test_data_path, _ = loader.make_dataset(server,dataset=exam_root, train_size=1)
            test_dataset = loader.CustomDataset(test_data_path[0], test_data_path[1], transform1,dataset=exam_root.lower().split('_')[0])
            tst_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)


    print('exam_root',exam_root)
    print(len(tst_loader))
    print(work_dir)
    prob_img_list = []
    input_img_list = []
    target_img_list = []
    image_name_list = []
    model.eval()

    with torch.no_grad():
        for i, (input, target,ori_img,image_name) in enumerate(tst_loader):

            input = input.cuda()
            target = target.cuda()

            try:
                output, _ = model(target)
            except:
                output, _,_ = model(target)

            # convert to prob

            pos_probs = torch.sigmoid(output)
            pos_probs = pos_probs.squeeze().cpu().numpy()
            input_ = ori_img.squeeze().cpu().numpy()
            target_ = target.squeeze().cpu().numpy()

            prob_img_list.append(pos_probs)
            input_img_list.append(input_)
            target_img_list.append(target_)

            image_name = image_name[0].split('/')[-1].split('.png')[0]
            image_name_list.append(image_name)


        print('end---------')
        return prob_img_list, input_img_list, target_img_list,image_name_list



def performance_by_slice(output_list, target_list,img_name_list):

    assert len(output_list) == len(target_list), 'not same list lenths'

    performance = {}
    for i in range(len(output_list)):
        preds =  output_list[i]
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
                               'img':img_name_list[i],
                               'acd_se': acd_se,
                               'asd_se': asd_se
                               }
        #'pixel': [gt_pixel, pred_pixel],

    return performance


def compute_overall_performance(collated_performance):
    confusion_matrix = np.zeros((4,))
    iou_sum = dice_sum = n_valid_slices = acd_sum = asd_sum = distanse_count = 0
    for res_slice in collated_performance.values():
        confusion_matrix += np.array(res_slice['cls'])
        if res_slice['gt'].sum() != 0: # consider only annotated slices
            iou_sum += res_slice['seg'][0]
            dice_sum += res_slice['seg'][1]
            n_valid_slices += 1

            if res_slice['acd_se'] ==None or res_slice['asd_se'] ==None:
                continue
            acd_sum += res_slice['acd_se']
            asd_sum += res_slice['asd_se']
            distanse_count+=1


    iou_mean = np.round(iou_sum / n_valid_slices, 3)
    dice_mean = np.round(dice_sum / n_valid_slices, 3)
    acd_se_mean = np.round(acd_sum / distanse_count, 3)
    asd_se_mean = np.round(asd_sum / distanse_count, 3)

    return { 'confusion_matrix': list(confusion_matrix),
             'slice_level_accuracy': (confusion_matrix[0] + confusion_matrix[2]) / confusion_matrix.sum(),
             'segmentation_performance': [iou_mean, dice_mean],'distance_performance[acd,asd]':[acd_se_mean,asd_se_mean]}


def compute_overall_pixel(collated_performance):

    confusion_matrix = np.zeros((4,))
    iou_sum = dice_sum = n_valid_slices = 0

    gt_pixel=[]
    pred_pixel=[]

    tp_pixel_gt = []
    tp_pixel_pred = []
    fp_pixel_gt = []
    fp_pixel_pred = []
    tn_pixel_gt = []
    tn_pixel_pred = []
    fn_pixel_gt = []
    fn_pixel_pred = []

    for res_exam in collated_performance.values():
        for res_slice in res_exam.values():

            gt_pixel.extend([res_slice['pixel_num'][0]])
            pred_pixel.extend([res_slice['pixel_num'][1]])

            #cls: [tp, fp, tn, fn]

            if res_slice['cls'][0] == 1:
                tp_pixel_gt.extend([res_slice['pixel_num'][0]])
                tp_pixel_pred.extend([res_slice['pixel_num'][1]])
            elif res_slice['cls'][1] == 1:
                fp_pixel_gt.extend([res_slice['pixel_num'][0]])
                fp_pixel_pred.extend([res_slice['pixel_num'][1]])
            elif res_slice['cls'][2] == 1:
                tn_pixel_gt.extend([res_slice['pixel_num'][0]])
                tn_pixel_pred.extend([res_slice['pixel_num'][1]])
            elif res_slice['cls'][3] == 1:
                fn_pixel_gt.extend([res_slice['pixel_num'][0]])
                fn_pixel_pred.extend([res_slice['pixel_num'][1]])

            confusion_matrix += np.array(res_slice['cls'])
            if res_slice['gt'].sum() != 0: # consider only annotated slices
                iou_sum += res_slice['seg'][0]
                dice_sum += res_slice['seg'][1]
                n_valid_slices += 1


    iou_mean = iou_sum / n_valid_slices
    dice_mean = dice_sum / n_valid_slices


    return {'confusion_matrix': list(confusion_matrix),
            'slice_level_accuracy': (confusion_matrix[0] + confusion_matrix[2]) / confusion_matrix.sum(),
            'segmentation_performance': [iou_mean, dice_mean],
            }


def load_model(network):

    if network == 'ae_v2':
        my_net = ae_lung(in_shape=(1, 256, 256))
    elif network == 'ae_tanh':
        my_net = ae_lung_tanh(in_shape=(1, 256, 256))

    else:
        raise ValueError('Not supported network.')

    model_name = str(my_net).split('(')[0]

    return my_net, model_name



def save_layer_fig(model,exam_id, org_input, org_target, prediction,
             slice_level_performance, result_dir):
    result_exam_dir = os.path.join(result_dir, exam_id)
    if not os.path.exists(result_exam_dir):
        os.makedirs(result_exam_dir)

    for name, param in model.named_parameters():
        print(name, '\t\t', param.shape)


def save_fig( org_input, org_target, prediction,
             slice_level_performance, result_dir):

    def _overlay_mask(img, mask, color='red'):

        # convert gray to color
        color_img = np.dstack([img, img, img])
        mask_idx = np.where(mask == 1)
        if color == 'red':
            color_img[mask_idx[0], mask_idx[1], :] = np.array([255,0,0])
        elif color == 'blue':
            color_img[mask_idx[0], mask_idx[1], :] = np.array([0,0,255])

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

        target_slice_pos_pixel =  target_slice.sum()
        target_slice_pos_pixel_rate = np.round(target_slice_pos_pixel/(256*256)*100,2)

        pred_slice_pos_pixel = pred_slice.sum()
        pred_slice_pos_pixel_rate = np.round(pred_slice_pos_pixel/(256*256) * 100, 2)


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
        try:
            ax[-1].set_title('IoU = {0:.4f} \n pred_pos_pixel = {1}({2}%) \n acd ={3:.3f} asd = {4:.3f}'.format(iou,
                                                                                                                pred_slice_pos_pixel,
                                                                                                                pred_slice_pos_pixel_rate,
                                                                                                                acd,
                                                                                                                asd))
        except:
            ax[-1].set_title('IoU = {0:.4f} \n pred_pos_pixel = {1}({2}%) \n acd =None asd = None'.format(iou,
                                                                                                          pred_slice_pos_pixel,
                                                                                                          pred_slice_pos_pixel_rate))

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


def seperate_dict(ori_dict, serch_list):
    new_dict = {}
    for i in serch_list:
        if i in ori_dict:  # key가 int인지 str인지 확인 필요
            new_dict[i] = ori_dict[i]
    return new_dict

def make_save_performance(collated_performance,level_id,dir_path,file_name,save_mode=False):
    sep_dict = seperate_dict(collated_performance, level_id)
    #overall_performance = compute_overall_performance(sep_dict)
    overall_performance = compute_overall_pixel(sep_dict)
    if save_mode:
        with open(os.path.join(dir_path, '{}_performance.json'.format(str(file_name))), 'w') as f:
            json.dump(overall_performance, f)
    return overall_performance



import decimal
class DecimalEncoder(json.JSONEncoder):
    def _iterencode(self, o, markers=None):
        if isinstance(o, decimal.Decimal):
            # wanted a simple yield str(o) in the next line,
            # but that would mean a yield on the line with super(...),
            # which wouldn't work (see my comment below), so...
            return (str(o) for o in [o])
        return super(DecimalEncoder, self)._iterencode(o, markers)


def cvae_mode(model,input,target,cvae_mode):

    if cvae_mode == 'target_coordconv':
        output, mu, logvar, core = model(target, target)
    elif cvae_mode == 'input_coordconv':
        output, mu, logvar, core = model(target, input)
    elif cvae_mode == 'input_plus_target_coordconv':
        input = torch.cat([input,target],1)
        output, mu, logvar, core = model(target, input)
    else:
        output, mu, logvar, core = model(target, input)


    return output, mu, logvar ,core

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='server_B')
    parser.add_argument('--exp', type=str)

    parser.add_argument('--test-root', default=['/data2/lung_segmentation_dataset/JSRT_dataset',
                                                '/data2/lung_segmentation_dataset/MC_modified_dataset',
                                                '/data2/lung_segmentation_dataset/SH_dataset'], nargs='+',type=str)

    parser.add_argument('--train-dataset', default='SH', help='JSRT_dataset,MC_dataset,SH_dataset')
    parser.add_argument('--test-dataset1', default='JSRT', help='JSRT_dataset,MC_dataset,SH_dataset')
    parser.add_argument('--test-dataset2', default='MC_modified', help='JSRT_dataset,MC_dataset,SH_dataset')


    parser.add_argument('--arch', default='unet', type=str)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--file-name', default='result_all_input', type=str)
    args = parser.parse_args()

    main_test(args=args)
    # test24_diceloss