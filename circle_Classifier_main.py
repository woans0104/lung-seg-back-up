import argparse
import torch.utils.data as data
import model
import ipdb
import os
import dataloader as loader
import numpy as np
from utils import Logger, AverageMeter, save_checkpoint ,draw_curve ,draw_curve_3,send_slack_message,str2bool
import time
import shutil
import pickle
import torch.optim as optim
from model import *
from losses import DiceLoss,tversky_loss, NLL_OHEM
import torch.backends.cudnn as cudnn
import torchvision as tv
from torchvision import models

parser = argparse.ArgumentParser()


# arguments for dataset
parser.add_argument('--trn-root', default='/data2/woans0104/circle_dataset',type=str)
parser.add_argument('--work-dir', default='/data1/JM/lung_segmentation')
parser.add_argument('--exp',default="test4", type=str)


parser.add_argument('--batch-size',default=4,type=int)


# arguments for model
parser.add_argument('--arch', default='unet', type=str)
parser.add_argument('--padding-size', default=1, type=int)
parser.add_argument('--batchnorm-momentum', default=0.1, type=float)
parser.add_argument('--coordconv-no', default=[9], nargs='+', type=int)
parser.add_argument('--radious',default=False,type=str2bool)


# arguments for optim & loss
parser.add_argument('--optim',default='sgd',choices=['adam','sgd'],type=str)
parser.add_argument('--eps',default=1e-08,type=float)
parser.add_argument('--weight-decay',default=1e-4,type=float)

parser.add_argument('--loss-function',default='bce',type=str)
parser.add_argument('--bce-weight', default=1, type=float)

parser.add_argument('--scheduler',default=True,type=str)
parser.add_argument('--initial-lr',default=0.1,type=float)
parser.add_argument('--lr-schedule', default=[100,120], nargs='+', type=int)
parser.add_argument('--gamma',default=0.1,type=float)

# arguments for dataset
parser.add_argument('--train-size',default=0.6,type=float)
parser.add_argument('--val-size',default=0.2,type=float)

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

    print(args.work_dir, args.exp)
    work_dir = os.path.join(args.work_dir, args.exp)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # copy this file to work dir to keep training configuration
    shutil.copy(__file__, os.path.join(work_dir, 'main.py'))
    with open(os.path.join(work_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
        
        

    # 1.dataset

    dataset_list = load_dataset(args.trn_root)

    train_image_paths, val_image_paths, test_image_paths = split_dataset(dataset_list,train_size=args.train_size,val_size=args.val_size)

    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomRotation(10),
        tv.transforms.RandomAffine(0,translate=(0.5,0.5), shear=10, scale=(0.8, 1.2)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.5],[0.5])
    ])

    train_set = loader.Circle_dataset(train_image_paths,transform=train_transforms)
    val_set = loader.Circle_dataset(val_image_paths, transform=train_transforms)
    test_set = loader.Circle_dataset(test_image_paths, transform=train_transforms)


    train_loader =  data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    #test set

    cir_test_dataset_list = load_dataset('/data2/woans0104/circle_test_dataset')
    cir_test_image_paths, cir_val_image_paths = split_dataset(cir_test_dataset_list, train_size=1)

    cir_test_set = loader.Circle_dataset(cir_test_image_paths, transform=train_transforms)
    cir_test_loader = data.DataLoader(cir_test_set, batch_size=1, shuffle=False, num_workers=4)



    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))
    tst_logger = Logger(os.path.join(work_dir, 'test.log'))
    cir_tst_logger = Logger(os.path.join(work_dir, 'cir_test.log'))


    # 3.model_select
    my_net, model_name = model_select(args.arch)


    # 4.gpu select
    my_net = nn.DataParallel(my_net).cuda()
    cudnn.benchmark = True

    # 5.optim

    if args.optim == 'adam':
        gen_optimizer = torch.optim.Adam(my_net.parameters(), lr=args.initial_lr, eps=args.eps)
    elif args.optim == 'sgd':
        gen_optimizer = torch.optim.SGD(my_net.parameters(), lr=args.initial_lr, momentum=0.9,weight_decay=args.weight_decay)



    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(gen_optimizer,
                                                  milestones=lr_schedule[:-1],
                                                  gamma=args.gamma)

    # 6.loss
    if args.loss_function == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.bce_weight])).cuda()
    elif args.loss_function == 'cle':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss_function == 'dice':
        criterion = DiceLoss().cuda()
    elif args.loss_function == 'mse':
        criterion = nn.MSELoss().cuda()



#####################################################################################

    # train

    send_slack_message(args.token, '#jm_private', '{} : starting_training'.format(args.exp))
    best_iou = 0
    try:
        if args.train_mode:
            for epoch in range(lr_schedule[-1]):

                train(my_net, train_loader, criterion,gen_optimizer, epoch,trn_logger)
                print('val acc **************************************************************')
                acc = test(my_net,val_loader, criterion, epoch, val_logger)
                print('JSRT_val acc **************************************************************')
                test(my_net, cir_test_loader, criterion, epoch, cir_tst_logger)

                lr_scheduler.step()

                is_best = acc > best_iou
                best_iou = max(acc, best_iou)
                checkpoint_filename = 'model_checkpoint_{:0>3}.pth'.format(epoch + 1)
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': my_net.state_dict(),
                                 'optimizer': gen_optimizer.state_dict()},
                                is_best,
                                work_dir,
                                checkpoint_filename)

        print("train end")
    except RuntimeError as e:
        send_slack_message(args.token, '#jm_private',
                       '-----------------------------------  error train : send to message JM  & Please send a kakao talk ----------------------------------------- \n error message : {}'.format(
                           e))
        import ipdb
        ipdb.set_trace()

    draw_curve(work_dir, trn_logger, val_logger,labelname='Acc')
    send_slack_message(args.token, '#jm_private', '{} : end_training'.format(args.exp))
    # validation
    if args.test_mode:
        print('Test mode ...')
        epoch=1
        test(my_net,test_loader, criterion, epoch, tst_logger)




def model_select(network):

    if network == 'All_conv':
        my_net = model.All_Convolutional()
    elif network == "resnet50":
        my_net = models.resnet50(pretrained=True)
        num_ftrs = my_net.fc.in_features
        my_net.fc = nn.Linear(num_ftrs, 4)
    elif network == "resnet18":
        my_net = models.resnet18(pretrained=True)
        num_ftrs = my_net.fc.in_features
        my_net.fc = nn.Linear(num_ftrs, 4)
    elif network == "SqueezeNet":
        my_net = models.squeezenet1_0(pretrained=True)
        my_net.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))
        my_net.num_classes = 4

    elif network == "Densenet121":
        my_net = models.densenet121(pretrained=True)
        num_ftrs = my_net.classifier.in_features
        my_net.classifier = nn.Linear(num_ftrs, 4)

    else:
        raise ValueError('Not supported network.')

    model_name = str(my_net).split('(')[0]

    return my_net, model_name


def train(model, trn_loader, criterion, optimizer,epoch,logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()


    # write your codes here
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(trn_loader):
        data_time.update(time.time() - end)

        input,target = input.cuda() , target.cuda()

        output = model(input)

        loss = criterion(output, target)

        losses.update(loss.item(), input.size(0))
        _, predicted = output.max(1)
        correct = predicted.eq(target).sum().item()
        correct= correct/target.size(0)
        acces.update(correct, target.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
            epoch, i, len(trn_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, acc=acces))


    logger.write([epoch, losses.avg, acces.avg])


def test(model, tst_loader, criterion,epoch,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()


    # write your codes here
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(tst_loader):


            input, target = input.cuda() ,target.cuda()

            output = model(input)
            loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))
            _, predicted = output.max(1)
            correct = predicted.eq(target).sum().item()

            """
            print('target',target)
            print('predicted',predicted)
            print('correct',correct)
            """
            acces.update(correct, target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()


    print(' * Loss {loss.avg:.3f}({loss.std:.3f}) Acc {acc.avg:.3f}({acc.std:.3f}) '.format(
           loss=losses, acc=acces))

    logger.write([epoch, losses.avg, acces.avg])

    return acces.avg




def load_dataset(data_dir):
    image_dir = []
    for (path, dir, files) in os.walk(data_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png':
                image_dir.append(os.path.join(path , filename))

    image_list = sorted(image_dir)

    return image_list

def split_dataset(image_paths,train_size,val_size=False):

    len_data = len(image_paths)
    indices_image = list(range(len_data))

    # np.random.seed(random_seed)
    np.random.shuffle(indices_image)

    image_paths = np.array(image_paths)


    if val_size:
        val_size = val_size

        train_image_no = indices_image[:int(len_data * train_size)]
        val_image_no = indices_image[int(len_data * train_size): int(len_data * (train_size + val_size))]
        test_image_no = indices_image[int(len_data * (train_size + val_size)):]

        train_image_paths = image_paths[train_image_no]

        val_image_paths = image_paths[val_image_no]

        test_image_paths = image_paths[test_image_no]


        return train_image_paths, val_image_paths, test_image_paths


    else:
        # no validation
        train_image_no = indices_image[:int(len_data * train_size)]
        test_image_no = indices_image[int(len_data * train_size):]

        train_image_paths = image_paths[train_image_no]
        test_image_paths = image_paths[test_image_no]


        print('end load data')
        return train_image_paths,test_image_paths



if __name__ == '__main__':
    main()