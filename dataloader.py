
import os
import glob
import random
import numpy as np
import ipdb

import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data.dataset import Dataset

import cv2





class Lung_Dataset(Dataset):

    def __init__(self,
                 image_paths,
                 target_paths,
                 transform,
                 aug_mode=False,
                 aug_range='aug6',
                 dataset='mc'):


        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms = transform
        self.aug_mode = aug_mode
        self.aug_range = aug_range
        self.dataset = dataset


    def aug(self, image, mask, aug_range):


        # transform pil image
        pil_image = transforms.ToPILImage()
        image = pil_image(image)
        mask = pil_image(mask)

        if aug_range == 'aug6':
            random_factor1 = random.random()
            random_factor2 = random.random()

            if random_factor1 > 0.5:

                # brightness
                brightness_factor = np.round((random.uniform(0.3, 1.1)), 1)
                image = transforms.functional.adjust_brightness(image, brightness_factor)

                if brightness_factor > 0.8:
                    # contrast
                    contrast_factor = np.round((random.uniform(0.8, 1.1)), 1)
                    image = transforms.functional.adjust_contrast(image, contrast_factor)
                else:
                    # contrast
                    contrast_factor = np.round((random.uniform(0.3, 1.1)), 1)
                    image = transforms.functional.adjust_contrast(image, contrast_factor)

            else:

                if random_factor2 > 0.5:
                    brightness_factor = np.round((random.uniform(0.3, 1.1)), 1)
                    image = transforms.functional.adjust_brightness(image, brightness_factor)

                else:
                    contrast_factor = np.round((random.uniform(0.3, 1.1)), 1)
                    image = transforms.functional.adjust_contrast(image, contrast_factor)

        elif aug_range == 'aug7':

            brightness_factor = random.uniform(0.4, 1.4)
            image = transforms.functional.adjust_brightness(image, brightness_factor)

            contrast_factor = random.uniform(0.4, 1.4)
            image = transforms.functional.adjust_contrast(image, contrast_factor)


        elif aug_range == 'aug9':

            brightness_factor = random.uniform(0.8, 1.2)
            image = transforms.functional.adjust_brightness(image, brightness_factor)

            contrast_factor = random.uniform(0.8, 1.2)
            image = transforms.functional.adjust_contrast(image, contrast_factor)

        elif aug_range == 'aug10':

            brightness_factor = random.uniform(0.6, 1.2)
            image = transforms.functional.adjust_brightness(image, brightness_factor)

            contrast_factor = random.uniform(0.6, 1.2)
            image = transforms.functional.adjust_contrast(image, contrast_factor)

        elif aug_range == 'aug11':

            #resized_crop = transforms.RandomResizedCrop(256, scale=(0.8,1.0))
            #color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4)
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)

            #transform = transforms.Compose([resized_crop, color_jitter])

            image = color_jitter(image)

            i, j, h, w = transforms.RandomResizedCrop.get_params(image,
                                                                 scale=(0.8,1.0),
                                                                 ratio=(0.9,1.1))
            image = transforms.functional.resized_crop(image, i, j, h, w, (256,256))
            mask = transforms.functional.resized_crop(mask, i, j, h, w, (256,256))

            image = np.array(image)
            mask = np.array(mask)

            return image, mask

        image = np.array(image)


        return image



    def __getitem__(self, index):

        # indexing test

        image_name = self.image_paths[index]
        target_name = self.target_paths[index]

        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(target_name, cv2.IMREAD_GRAYSCALE)

        # jsrt dataset= invert
        if self.dataset.lower() == 'jsrt':
            image = cv2.bitwise_not(image)

        image = cv2.equalizeHist(image)

        # cv2 resize
        #image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        #mask = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, dsize=(256, 256))
        mask = cv2.resize(mask, dsize=(256, 256))

        # aug
        if self.aug_mode:
            image, mask = self.aug(image, mask, self.aug_range)

        image_tensor = self.transforms(image)

        if np.max(mask) > 1:
            mask = mask / 255
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0

        mask = mask * 255
        mask = np.expand_dims(mask, -1)
        mask = np.array(mask, dtype=np.uint8)

        assert len(set(mask.flatten())) == 2, 'mask label is wrong'

        toTensor = transforms.ToTensor()
        mask_tensor = toTensor(mask)


        return image_tensor, mask_tensor, image, image_name

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)




def dataset_condition(trainset_condition):
    dataset = {
        'JSRT': ['MC_modified', 'SH'],
        'MC_modified': ['JSRT', 'SH'],
        'SH': ['JSRT', 'MC_modified']
    }

    if trainset_condition in dataset.keys():
        print('*' * 50)
        print('train dataset : ', trainset_condition)
        print('test dataset1 : ', dataset[trainset_condition][0])
        print('test dataset2 : ', dataset[trainset_condition][1])
        print('*' * 50)

        train_datset = trainset_condition
        test_dataset1 = dataset[trainset_condition][0]
        test_dataset2 = dataset[trainset_condition][1]

        return train_datset, test_dataset1, test_dataset2

    else:
        import ipdb;
        ipdb.set_trace()




def get_loader(server, dataset, train_size, batch_size, aug_mode, aug_range, work_dir=None):

    # transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])


    if train_size != 1:
        train_image_path, train_label_path, test_image_path, test_label_path = load_data_path(server, dataset,
                                                                                              train_size=train_size)

        np.save(os.path.join(work_dir, '{}_test_path.npy'.format(dataset)),
                [test_image_path, test_label_path])  ########



        train_dataset = Lung_Dataset(train_image_path, train_label_path, transform, aug_mode=aug_mode,aug_range=aug_range,
                                            dataset=dataset)
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       num_workers=4)

        test_dataset = Lung_Dataset(test_image_path, test_label_path, transform,dataset=dataset)

        test_loader = data.DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0)

        return train_loader, test_loader

    else: # train_size == 1

        train_image_path, train_label_path = load_data_path(server, dataset, train_size=train_size)

        train_dataset = Lung_Dataset(train_image_path, train_label_path, transform, aug_mode=aug_mode,
                                     aug_range=aug_range,dataset=dataset)

        train_loader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       num_workers=4)


        return train_loader , train_loader






def load_data_path(server, dataset, train_size):


    def read_data(data_folder):

        valid_exts = ['.jpg', '.gif', '.png', '.tga', '.jpeg']

        data_paths = []
        for f in data_folder:
            ext = os.path.splitext(f)[1]

            if ext.lower() not in valid_exts:
                continue
            data_paths.append(f)

        return data_paths

    def match_data_path(img_path, target_path):

        img_path = np.array(sorted(img_path))
        target_path = np.array(sorted(target_path))

        # list name sort
        imgName_li = []
        for i in range(len(img_path)):
            img_name = img_path[i].split('/')[-1].split('.')[0]
            imgName_li.append(img_name)

        total_img_li = []
        for i in range(len(target_path)):
            try:
                img_name = target_path[i].split('/')[-1].split('.')[0].split('_mask')[0]

                idx = np.where(imgName_li == np.array(img_name))[0][0]

                total_img_li.append(img_path[idx])

            except IndexError:
                continue

        return total_img_li, target_path


    dataset = dataset + '_dataset'

    #####################################################################################################################
    if server == 'server_A':
        image_folder = sorted(glob.glob("/data2/woans0104/lung_segmentation_dataset/{}/image/*".format(dataset)))
        target_folder = sorted(glob.glob("/data2/woans0104/lung_segmentation_dataset/{}/label/*".format(dataset)))
    elif server == 'server_B':
        image_folder = sorted(glob.glob("/data2/lung_segmentation_dataset/{}/image/*".format(dataset)))
        target_folder = sorted(glob.glob("/data2/lung_segmentation_dataset/{}/label/*".format(dataset)))
    elif server == 'server_D':
        image_folder = sorted(glob.glob('/daintlab/data/lung_segmentation_dataset/{}/image/*'.format(dataset)))
        target_folder = sorted(glob.glob('/daintlab/data/lung_segmentation_dataset/{}/label/*'.format(dataset)))
    #####################################################################################################################

    image_paths =read_data(image_folder)
    target_paths = read_data(target_folder)

    if len(image_paths) != len(target_paths):
        image_paths, target_paths = match_data_path(image_paths, target_paths)

    assert len(image_paths) == len(target_paths), print(target_paths)#'different length img & mask'

    # last sort
    image_paths = sorted(image_paths)
    target_paths = sorted(target_paths)

    len_data = len(image_paths)
    indices_image = list(range(len_data))


    np.random.shuffle(indices_image)

    image_paths = np.array(image_paths)
    target_paths = np.array(target_paths)

    train_image_no = indices_image[:int(len_data * train_size)]
    test_image_no = indices_image[int(len_data * train_size):]

    train_image_paths = image_paths[train_image_no]
    train_mask_paths = target_paths[train_image_no]

    test_image_paths = image_paths[test_image_no]
    test_mask_paths = target_paths[test_image_no]


    if train_size ==1:
        return train_image_paths, train_mask_paths
    else :
        return train_image_paths, train_mask_paths, test_image_paths, test_mask_paths







