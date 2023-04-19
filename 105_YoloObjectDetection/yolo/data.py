"""
CS 6384 Homework 5 Programming
Implement the __getitem__() function in this python script
"""
import os
import glob
import math

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision.io


# The dataset class
class CrackerBox(data.Dataset):
    def __init__(self, image_set='train', data_path='data'):

        self.name = 'cracker_box_' + image_set
        self.image_set = image_set
        self.data_path = data_path
        self.classes = ('__background__', 'cracker_box')
        self.width = 640
        self.height = 480
        self.yolo_image_size = 448
        self.scale_width = self.yolo_image_size / self.width
        self.scale_height = self.yolo_image_size / self.height
        self.yolo_grid_num = 7
        self.yolo_grid_size = self.yolo_image_size / self.yolo_grid_num
        # split images into training set and validation set
        self.gt_files_train, self.gt_files_val = self.list_dataset()
        # the pixel mean for normalization
        self.pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)

        # training set
        if image_set == 'train':
            self.size = len(self.gt_files_train)
            self.gt_paths = self.gt_files_train
            print('%d images for training' % self.size)
        else:
            # validation set
            self.size = len(self.gt_files_val)
            self.gt_paths = self.gt_files_val
            print('%d images for validation' % self.size)

    # list the ground truth annotation files
    # use the first 100 images for training
    def list_dataset(self):

        filename = os.path.join(self.data_path, '*.txt')
        gt_files = sorted(glob.glob(filename))

        gt_files_train = gt_files[:100]
        gt_files_val = gt_files[100:]

        return gt_files_train, gt_files_val

    # TODO: implement this function
    def __getitem__(self, idx):

        # gt file
        filename_gt = self.gt_paths[idx]

        ### ADD YOUR CODE HERE ###
        filename = filename_gt.split('-')[0]
        filename_jpg = os.path.join(filename + '.jpg')
        img_size = self.yolo_image_size

        img_orig = cv2.imread(filename_jpg)
        img_resized = cv2.resize(img_orig, [img_size, img_size])
        img_normalize = img_resized - self.pixel_mean
        img_normalize = img_normalize / 255.0
        # image_blob = torchvision.transforms.functional.to_tensor(img_resized) # Incase below throws error
        image_blob = torchvision.transforms.ToTensor()(img_normalize)
        img_orig_height, img_orig_width = img_orig.shape[:-1]

        # Using torchvision transforms has an issue of BGR to RGB
        # Ref: https://www.programcreek.com/python/example/104831/torchvision.transforms.ToPILImage
        # import torchvision.transforms as T
        # img_orig = torchvision.io.read_image(filename_jpg)
        # img_height, img_width = img_orig.shape[1:]
        # to_bgr_transform = T.Lambda(lambda xd: xd[[2, 1, 0]])
        # pixel_mean = self.pixel_mean.T
        # transforms = T.Compose([
        #     T.ToPILImage(),
        #     T.Resize([img_size, img_size]),
        #     T.ToTensor(),
        #     to_bgr_transform,
        #     T.Normalize(pixel_mean, [1.0, 1.0, 1.0]),
        #
        # ])
        # image_blob = transforms(img_orig)
        # torchvision.io.write_jpeg(T.ConvertImageDtype(torch.uint8)(image_blob), "image_blob.jpg")

        # we have to do for each bounding box in case of multiple classes.
        y_scale = self.yolo_image_size / img_orig_height
        x_scale = self.yolo_image_size / img_orig_width
        with open(filename_gt, 'r') as f:
            gt = f.readline()
            bbox = np.array(gt.split()).astype(np.float64)
        [x1, y1, x2, y2] = bbox

        y1 *= y_scale
        y2 *= y_scale

        x1 *= x_scale
        x2 *= x_scale

        cy = (y1 + y2) / 2
        cx = (x1 + x2) / 2
        cy = cy / self.yolo_grid_size
        cx = cx / self.yolo_grid_size

        img_height = abs(y2 - y1) / self.yolo_image_size
        img_width = abs(x2 - x1) / self.yolo_image_size

        gt_box_blob = np.zeros([5, 7, 7])
        gt_mask_blob = np.zeros([7, 7])
        gt_i = math.floor(cx)
        gt_j = math.floor(cy)
        # In case of multiple classes (more than 1) need to use the one hot encoding for the last term
        gt_box_blob[:, gt_j, gt_i] = [cx - gt_i, cy - gt_j, img_width, img_height, 1]
        gt_mask_blob[gt_j, gt_i] = 1

        # this is the sample dictionary to be returned from this function
        sample = {'image': image_blob,
                  'gt_box': gt_box_blob,
                  'gt_mask': gt_mask_blob}

        return sample

    # len of the dataset
    def __len__(self):
        return self.size


# draw grid on images for visualization
def draw_grid(image, line_space=64):
    H, W = image.shape[:2]
    image[0:H:line_space, :] = [255, 255, 0]
    image[:, 0:W:line_space] = [255, 255, 0]


# the main function for testing
if __name__ == '__main__':
    dataset_train = CrackerBox('train')
    dataset_val = CrackerBox('val')

    # dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)

    # visualize the training data
    for i, sample in enumerate(train_loader):
        image = sample['image'][0].numpy().transpose((1, 2, 0))
        gt_box = sample['gt_box'][0].numpy()
        gt_mask = sample['gt_mask'][0].numpy()

        y, x = np.where(gt_mask == 1)
        cx = gt_box[0, y, x] * dataset_train.yolo_grid_size + x * dataset_train.yolo_grid_size
        cy = gt_box[1, y, x] * dataset_train.yolo_grid_size + y * dataset_train.yolo_grid_size
        w = gt_box[2, y, x] * dataset_train.yolo_image_size
        h = gt_box[3, y, x] * dataset_train.yolo_image_size

        x1 = cx - w * 0.5
        x2 = cx + w * 0.5
        y1 = cy - h * 0.5
        y2 = cy + h * 0.5

        print(image.shape, gt_box.shape)

        # visualization
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        im = image * 255.0 + dataset_train.pixel_mean
        im = im.astype(np.uint8)
        plt.imshow(im[:, :, (2, 1, 0)])
        plt.title('input image (448x448)', fontsize=16)

        ax = fig.add_subplot(1, 3, 2)
        draw_grid(im)
        plt.imshow(im[:, :, (2, 1, 0)])
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor="none")
        ax.add_patch(rect)
        plt.plot(cx, cy, 'ro', markersize=12)
        plt.title('Ground truth bounding box in YOLO format', fontsize=16)

        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(gt_mask)
        plt.title('Ground truth mask in YOLO format (7x7)', fontsize=16)
        plt.show()
