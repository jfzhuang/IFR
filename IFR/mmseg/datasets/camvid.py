import os.path as osp
import tempfile
import random

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class CamVidDataset(CustomDataset):
    CLASSES = (
        'road',
        'sidewalk',
        'building',
        'wall',
        'fence',
        'pole',
        'traffic light',
        'traffic sign',
        'vegetation',
        'terrain',
        'sky',
    )

    PALETTE = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
    ]

    def __init__(self, **kwargs):
        super(CamVidDataset, self).__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)


@DATASETS.register_module()
class CamVidSemiDataset(CustomDataset):
    CLASSES = (
        'road',
        'sidewalk',
        'building',
        'wall',
        'fence',
        'pole',
        'traffic light',
        'traffic sign',
        'vegetation',
        'terrain',
        'sky',
    )

    PALETTE = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
    ]

    def __init__(
        self,
        pipeline,
        img_dir,
        ann_dir,
        img_suffix='.png',
        seg_map_suffix='.png',
        split=None,
        split_unlabeled=None,
        data_root=None,
        ignore_index=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        clip_length=30,
    ):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.split_unlabeled = split_unlabeled
        self.data_root = data_root
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.clip_length = clip_length

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)
            if not (self.split_unlabeled is None or osp.isabs(self.split_unlabeled)):
                self.split_unlabeled = osp.join(self.data_root, self.split_unlabeled)

        # load annotations
        self.video_infos_labeled, self.video_infos_unlabeled = self.load_annotations(
            self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split, self.split_unlabeled
        )

    def __len__(self):
        """Total number of samples of data."""
        return len(self.video_infos_labeled)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split, split_unlabeled):
        video_infos_labeled = []
        with open(split) as f:
            lines = f.readlines()

        for i in range(len(lines) // self.clip_length):
            video_lines = lines[i * self.clip_length : (i + 1) * self.clip_length]
            img_infos = []
            for line in video_lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix, ann=img_name + seg_map_suffix)
                img_infos.append(img_info)
            video_infos_labeled.append(img_infos)

        print_log(f'Loaded {len(video_infos_labeled)} labeled clips', logger=get_root_logger())

        video_infos_unlabeled = []
        with open(split_unlabeled) as f:
            lines = f.readlines()

        for i in range(len(lines) // self.clip_length):
            video_lines = lines[i * self.clip_length : (i + 1) * self.clip_length]
            img_infos = []
            for line in video_lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix, ann=img_name + seg_map_suffix)
                img_infos.append(img_info)
            video_infos_unlabeled.append(img_infos)

        print_log(f'Loaded {len(video_infos_unlabeled)} unlabeled clips', logger=get_root_logger())
        return video_infos_labeled, video_infos_unlabeled

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir

    def __getitem__(self, video_idx_0):
        idx_v0_0 = 29
        idx_v0_1 = random.choice([i for i in range(self.clip_length) if i != idx_v0_0])

        video_idx_1 = random.randint(0, len(self.video_infos_unlabeled) - 1)
        idx_list = [i for i in range(self.clip_length)]
        random.shuffle(idx_list)
        idx_v1_0, idx_v1_1 = idx_list[:2]

        return self.prepare_train_img(video_idx_0, idx_v0_0, idx_v0_1, video_idx_1, idx_v1_0, idx_v1_1)

    def prepare_train_img(self, video_idx_0, idx_v0_0, idx_v0_1, video_idx_1, idx_v1_0, idx_v1_1):
        img_info_v0_0 = self.video_infos_labeled[video_idx_0][idx_v0_0]
        img_info_v0_1 = self.video_infos_labeled[video_idx_0][idx_v0_1]
        img_info_v1_0 = self.video_infos_unlabeled[video_idx_1][idx_v1_0]
        img_info_v1_1 = self.video_infos_unlabeled[video_idx_1][idx_v1_1]
        results = dict(
            img_info_v0_0=img_info_v0_0,
            img_info_v0_1=img_info_v0_1,
            img_info_v1_0=img_info_v1_0,
            img_info_v1_1=img_info_v1_1,
        )
        self.pre_pipeline(results)
        return self.pipeline(results)
