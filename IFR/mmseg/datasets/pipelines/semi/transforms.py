import mmcv
import numpy as np
from numpy import random
from mmseg.datasets.builder import PIPELINES
from mmcv.utils import deprecated_api_warning


@PIPELINES.register_module()
class RandomCrop_Semi(object):
    def __init__(self, crop_size, cat_max_ratio=1.0, ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        img = results['img_v0_0']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.0:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)
        results['img_v0_0'] = self.crop(results['img_v0_0'], crop_bbox)
        results['img_v0_1'] = self.crop(results['img_v0_1'], crop_bbox)
        results['gt'] = self.crop(results['gt'], crop_bbox)

        crop_bbox = self.get_crop_bbox(img)
        results['img_v1_0'] = self.crop(results['img_v1_0'], crop_bbox)
        results['img_v1_1'] = self.crop(results['img_v1_1'], crop_bbox)

        results['img_shape'] = results['img_v0_0'][0].shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class RandomFlip_Semi(object):
    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip_Semi')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        if 'flip_v0_0' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip_v0_0'] = flip
        if 'flip_v0_1' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip_v0_1'] = flip
        if 'flip_v1_0' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip_v1_0'] = flip
        if 'flip_v1_1' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip_v1_1'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction

        if results['flip_v0_0']:
            results['gt'] = mmcv.imflip(results['gt'], direction=results['flip_direction'])
            results['img_v0_0'] = mmcv.imflip(results['img_v0_0'], direction=results['flip_direction'])
        if results['flip_v0_1']:
            results['img_v0_1'] = mmcv.imflip(results['img_v0_1'], direction=results['flip_direction'])
        if results['flip_v1_0']:
            results['img_v1_0'] = mmcv.imflip(results['img_v1_0'], direction=results['flip_direction'])
        if results['flip_v1_1']:
            results['img_v1_1'] = mmcv.imflip(results['img_v1_1'], direction=results['flip_direction'])

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class PhotoMetricDistortion_Semi(object):
    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(img, alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1], alpha=random.uniform(self.saturation_lower, self.saturation_upper)
            )
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        results['img_v0_1_s'] = results['img_v0_1'].copy()
        results['img_v1_1_s'] = results['img_v1_1'].copy()

        img = results['img_v0_1_s']
        img = self.brightness(img)
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)
        img = self.saturation(img)
        img = self.hue(img)
        if mode == 0:
            img = self.contrast(img)
        results['img_v0_1_s'] = img

        img = results['img_v1_1_s']
        img = self.brightness(img)
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)
        img = self.saturation(img)
        img = self.hue(img)
        if mode == 0:
            img = self.contrast(img)
        results['img_v1_1_s'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(brightness_delta={self.brightness_delta}, '
            f'contrast_range=({self.contrast_lower}, '
            f'{self.contrast_upper}), '
            f'saturation_range=({self.saturation_lower}, '
            f'{self.saturation_upper}), '
            f'hue_delta={self.hue_delta})'
        )
        return repr_str


@PIPELINES.register_module()
class Normalize_Semi(object):
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results['img_v0_0'] = mmcv.imnormalize(results['img_v0_0'], self.mean, self.std, self.to_rgb)
        results['img_v0_1'] = mmcv.imnormalize(results['img_v0_1'], self.mean, self.std, self.to_rgb)
        results['img_v0_1_s'] = mmcv.imnormalize(results['img_v0_1_s'], self.mean, self.std, self.to_rgb)
        results['img_v1_0'] = mmcv.imnormalize(results['img_v1_0'], self.mean, self.std, self.to_rgb)
        results['img_v1_1'] = mmcv.imnormalize(results['img_v1_1'], self.mean, self.std, self.to_rgb)
        results['img_v1_1_s'] = mmcv.imnormalize(results['img_v1_1_s'], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' f'{self.to_rgb})'
        return repr_str
