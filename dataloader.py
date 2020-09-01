from torch.utils.data.dataset import Dataset
import numpy as np
import torchvision.transforms as transforms
import random
import glob
import os
import cv2
import torch.utils.data as data
import ipdb
from PIL import Image


# from scipy.misc import imread, imresize,imsave


class CustomDataset(Dataset):

    def __init__(self, image_paths, target_paths, transform1, arg_mode=False, arg_thres=0, arg_range='arg1',
                 dataset='mc'):  # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms1 = transform1
        self.arg_mode = arg_mode
        self.arg_thres = 0.5
        self.arg_range = arg_range
        self.dataset = dataset

    def arg(self, image, arg_thres, arg_range):
        """
        # Resize

        resize = transforms.Resize(size=(256, 256))
        image = resize(image)
        mask = resize(mask)
        """
        # transform pil image
        pil_image = transforms.ToPILImage()
        image = pil_image(image)

        # from PIL import Image
        # image = Image.fromarray(image)
        # print(type(image))
        """
        # Random horizontal flipping
        horizontal=random.random()
        #print('horizontal flipping', horizontal)
        if horizontal > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Random vertical flipping
        vertical=random.random()
        #print('vertical flipping', vertical)
        if vertical > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        # Random rotation
        Rrotation = random.random()
        #print('Rrotation', Rrotation)
        if Rrotation > 0.5:
            angle = random.randint(-30, 30)
            image = transforms.functional.rotate(image,angle)
            mask = transforms.functional.rotate(mask,angle)
        """

        brightness = random.random()
        contrast = random.random()

        if arg_range == 'arg1':
            import ipdb;
            ipdb.set_trace()
            if brightness > arg_thres and contrast > arg_thres:

                brightness_factor = random.uniform(0.5, 1.0)
                image = transforms.functional.adjust_brightness(image, brightness_factor)

                contrast_factor = random.uniform(0.8, 1.3)
                image = transforms.functional.adjust_contrast(image, contrast_factor)


            else:
                if brightness > arg_thres:
                    brightness_factor = random.uniform(0.3, 1.1)
                    image = transforms.functional.adjust_brightness(image, brightness_factor)

                if contrast > arg_thres:
                    contrast_factor = random.uniform(0.7, 1.0)
                    image = transforms.functional.adjust_contrast(image, contrast_factor)

        elif arg_range == 'arg2':
            import ipdb;
            ipdb.set_trace()
            if brightness > arg_thres and contrast > arg_thres:

                brightness_factor = random.uniform(0.5, 1.0)
                image = transforms.functional.adjust_brightness(image, brightness_factor)

                contrast_factor = random.uniform(0.3, 0.8)
                image = transforms.functional.adjust_contrast(image, contrast_factor)


            else:
                if brightness > arg_thres:
                    brightness_factor = random.uniform(0.3, 1.1)
                    image = transforms.functional.adjust_brightness(image, brightness_factor)

                if contrast > arg_thres:
                    contrast_factor = random.uniform(0.3, 0.8)
                    image = transforms.functional.adjust_contrast(image, contrast_factor)

        elif arg_range == 'arg3':

            # brightness
            # brightness_factor = np.round((random.uniform(0.3, 1.1)), 1)
            brightness_factor = random.uniform(0.3, 1.1)
            image1 = transforms.functional.adjust_brightness(image, brightness_factor)

            # contrast
            # contrast_factor = np.round((random.uniform(0.3, 1.1)), 1)
            contrast_factor = random.uniform(0.3, 1.1)
            image = transforms.functional.adjust_contrast(image1, contrast_factor)

            # #brightness
            # brightness_factor = random.uniform(0.6, 1.1)
            # image1 = transforms.functional.adjust_brightness(image, brightness_factor)
            #
            # #contrast
            # contrast_factor = random.uniform(0.6, 1.1)
            # image = transforms.functional.adjust_contrast(image1, contrast_factor)



        elif arg_range == 'arg4':
            random_factor1 = random.random()
            random_factor2 = random.random()

            if random_factor1 > 0.5:

                # brightness
                brightness_factor = np.round((random.uniform(0.3, 1.1)), 1)
                image1 = transforms.functional.adjust_brightness(image, brightness_factor)

                # contrast
                contrast_factor = np.round((random.uniform(0.3, 1.1)), 1)
                image = transforms.functional.adjust_contrast(image1, contrast_factor)


            else:

                if random_factor2 > 0.5:
                    brightness_factor = np.round((random.uniform(0.3, 1.1)), 1)
                    image = transforms.functional.adjust_brightness(image, brightness_factor)

                else:
                    contrast_factor = np.round((random.uniform(0.3, 1.1)), 1)
                    image = transforms.functional.adjust_contrast(image, contrast_factor)


        elif arg_range == 'arg5':
            random_factor1 = random.random()
            random_factor2 = random.random()
            brightness_factor = 0.5

            if random_factor1 > 0.5:
                # brightness
                brightness_factor = np.round((random.uniform(0.3, 1.1)), 1)
                image = transforms.functional.adjust_brightness(image, brightness_factor)

            if random_factor2 > 0.5:

                if brightness_factor > 0.8:
                    # contrast
                    # print(brightness_factor)
                    contrast_factor = np.round((random.uniform(0.8, 1.1)), 1)
                    image = transforms.functional.adjust_contrast(image, contrast_factor)
                else:
                    # contrast
                    contrast_factor = np.round((random.uniform(0.3, 1.1)), 1)
                    image = transforms.functional.adjust_contrast(image, contrast_factor)

        elif arg_range == 'arg6':
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


        elif arg_range == 'arg7':

            # original img 0.5 augmentation 0.5

            random_factor0 = random.random()
            random_factor1 = random.random()
            random_factor2 = random.random()


            if random_factor0 > arg_thres:
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



        elif arg_range == 'arg8':

            # min range up , max range up

            random_factor1 = random.random()
            random_factor2 = random.random()

            if random_factor1 > 0.5:

                # brightness
                brightness_factor = np.round((random.uniform(0.5, 1.2)), 1)
                image = transforms.functional.adjust_brightness(image, brightness_factor)

                if brightness_factor > 0.8:
                    # contrast
                    contrast_factor = np.round((random.uniform(0.8, 1.2)), 1)
                    image = transforms.functional.adjust_contrast(image, contrast_factor)
                else:
                    # contrast
                    contrast_factor = np.round((random.uniform(0.5, 1.2)), 1)
                    image = transforms.functional.adjust_contrast(image, contrast_factor)

            else:

                if random_factor2 > 0.5:
                    brightness_factor = np.round((random.uniform(0.5, 1.2)), 1)
                    image = transforms.functional.adjust_brightness(image, brightness_factor)

                else:
                    contrast_factor = np.round((random.uniform(0.5, 1.2)), 1)
                    image = transforms.functional.adjust_contrast(image, contrast_factor)






        else:
            import ipdb;
            ipdb.set_trace()

        # Transform to tensor
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

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
            # print('invert')

        # cv2 resize
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

        # arg
        if self.arg_mode:
            image = self.arg(image, self.arg_thres, self.arg_range)

        t_image = self.transforms1(image)

        if np.max(mask) > 1:
            mask = mask / 255
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0

        mask = np.expand_dims(mask, -1)
        mask = mask * 255
        mask = np.array(mask, dtype=np.uint8)

        assert len(set(mask.flatten())) == 2, 'mask label is wrong'

        totensor = transforms.ToTensor()
        t_mask = totensor(mask)

        return t_image, t_mask, image, image_name

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)


def make_SHdataset(folder_data, folder_mask):
    folder_data = np.array(sorted(folder_data))
    folder_mask = np.array(sorted(folder_mask))

    # list name sort
    list_dataName = []
    for i in range(len(folder_data)):
        data_name = folder_data[i].split('/')[-1].split('.')[0]
        list_dataName.append(data_name)

    listdata = []
    for i in folder_mask:
        try:
            mask_name = i.split('/')[-1].split('.')[0].split('_mask')[0]
            list_dataName = np.array(list_dataName)

            idx = np.where(list_dataName == mask_name)[0][0]

            listdata.append(folder_data[idx])
        except IndexError:
            continue

    return listdata, folder_mask


def make_dataset(server, dataset, train_size):
    # simple division : train_size_6,val_size_2,test_size_2
    # if valsize == false : train_size_7, test_size_3
    print('now dataset', dataset)
    # MC_dataset : not modified

    valid_exts = ['.jpg', '.gif', '.png', '.tga', '.jpeg']  # 이 확장자들만 불러오겠다.

    if server == 'server_A':
        image_folder = sorted(glob.glob("/data2/woans0104/lung_segmentation_dataset/{}/image/*".format(dataset)))
        target_folder = sorted(glob.glob("/data2/woans0104/lung_segmentation_dataset/{}/label/*".format(dataset)))
    elif server == 'server_B':
        image_folder = sorted(glob.glob("/data2/lung_segmentation_dataset/{}/image/*".format(dataset)))
        target_folder = sorted(glob.glob("/data2/lung_segmentation_dataset/{}/label/*".format(dataset)))

    # 폴더 안의 확장자들만 가져오기

    image_paths = []
    for f in image_folder:
        ext = os.path.splitext(f)[1]  # 확장자만 가져오기

        if ext.lower() not in valid_exts:
            continue
        image_paths.append(f)

    target_paths = []
    for f in target_folder:
        ext = os.path.splitext(f)[1]  # 확장자만 가져오기
        if ext.lower() not in valid_exts:
            continue
        target_paths.append(f)

    if len(image_paths) != len(target_paths):
        image_paths, target_paths = make_SHdataset(image_paths, target_paths)

    # last sort
    image_paths = sorted(image_paths)
    target_paths = sorted(target_paths)

    assert len(image_paths) == len(target_paths), 'different length img & mask'

    # random_seed = 10

    len_data = len(image_paths)
    indices_image = list(range(len_data))

    # np.random.seed(random_seed)
    np.random.shuffle(indices_image)

    image_paths = np.array(image_paths)
    target_paths = np.array(target_paths)

    train_image_no = indices_image[:int(len_data * train_size)]
    test_image_no = indices_image[int(len_data * train_size):]

    train_image_paths = image_paths[train_image_no]
    train_mask_paths = target_paths[train_image_no]

    test_image_paths = image_paths[test_image_no]
    test_mask_paths = target_paths[test_image_no]

    print('end load data')
    return [train_image_paths, train_mask_paths], [test_image_paths, test_mask_paths]







