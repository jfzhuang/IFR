import os
import sys
import cv2
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


class cityscapes_video_dataset_accel(Dataset):
    def __init__(self, split='train', partition=30):
        self.data_path = '/home/zhuangjiafan/semi-vss/Accel/data'
        self.im_path = os.path.join(self.data_path, 'cityscapes', 'leftImg8bit_sequence_down_2x', split)
        self.gt_path = os.path.join(self.data_path, 'cityscapes', 'gtFine_down_2x', split)
        self.split = split
        self.partition = partition
        self.crop_size = (256, 512)
        self.get_list()

        self.mean = np.array([123.675, 116.28, 103.53])
        self.mean = np.expand_dims(np.expand_dims(self.mean, axis=1), axis=1)
        self.std = np.array([58.395, 57.12, 57.375])
        self.std = np.expand_dims(np.expand_dims(self.std, axis=1), axis=1)
        print('load {} clips'.format(len(self)))

    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'list', 'train_sup_1-{}.txt'.format(self.partition))
        else:
            list_path = os.path.join(self.data_path, 'list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)

            frame_id = int(line[-6:])
            frame_prefix = line[:-7]
            tmp = []
            for i in range(5):
                name = '{}_{:06d}'.format(frame_prefix, frame_id - 4 + i)
                tmp.append(name)
            self.im_name.append(tmp)

    def __len__(self):
        return len(self.gt_name)

    def transform(self, im_list):
        im_seg_list = []
        im_flow_list = []
        for i in range(len(im_list)):
            im = im_list[i]
            im_flow = im.copy()
            im_flow = im_flow.transpose((2, 0, 1))
            im_flow = im_flow.astype(np.float32) / 255.0
            im_flow_list.append(im_flow)

            im_seg = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_seg = im_seg.transpose((2, 0, 1))
            im_seg = (im_seg - self.mean) / self.std
            im_seg_list.append(im_seg)

        return im_seg_list, im_flow_list

    def random_crop(self, im_seg_list, im_flow_list, gt):
        h, w = gt.shape
        crop_h, crop_w = self.crop_size
        start_h = random.randint(0, h - crop_h - 1)
        start_w = random.randint(0, w - crop_w - 1)
        for i in range(len(im_seg_list)):
            im_seg_list[i] = im_seg_list[i][:, start_h : start_h + crop_h, start_w : start_w + crop_w]
            im_flow_list[i] = im_flow_list[i][:, start_h : start_h + crop_h, start_w : start_w + crop_w]
        gt = gt[start_h : start_h + crop_h, start_w : start_w + crop_w]
        return im_seg_list, im_flow_list, gt

    def __getitem__(self, idx):
        im_name_list = self.im_name[idx]
        im_list = []
        for i in range(len(im_name_list)):
            name = im_name_list[i]
            im = cv2.imread(os.path.join(self.im_path, '{}_leftImg8bit.png'.format(name)))
            im_list.append(im)
        gt = cv2.imread(os.path.join(self.gt_path, '{}_gtFine_labelTrainIds.png'.format(name)), 0)

        im_seg_list, im_flow_list = self.transform(im_list)
        if self.split == 'train':
            im_seg_list, im_flow_list, gt = self.random_crop(im_seg_list, im_flow_list, gt)

        for i in range(len(im_list)):
            im_seg_list[i] = torch.from_numpy(im_seg_list[i]).float().unsqueeze(1)
            im_flow_list[i] = torch.from_numpy(im_flow_list[i]).float().unsqueeze(1)
        im_seg_list = torch.cat(im_seg_list, dim=1)
        im_flow_list = torch.cat(im_flow_list, dim=1)
        gt = torch.from_numpy(gt.astype(np.int64)).long()

        return im_seg_list, im_flow_list, gt


if __name__ == '__main__':
    dataset = cityscapes_video_dataset_accel(split='val')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
    for i, data in enumerate(dataloader):
        im_seg_list, im_flow_list, gt = data
        print('{}/{}'.format(i, len(dataloader)), im_seg_list.shape, im_flow_list.shape, gt.shape)
