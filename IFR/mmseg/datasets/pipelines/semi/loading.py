import os.path as osp

import mmcv
import numpy as np

from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile_Semi(object):
    def __init__(
        self, to_float32=False, color_type='color', file_client_args=dict(backend='disk'), imdecode_backend='cv2'
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filenames_v0_0 = osp.join(results['img_prefix'], results['img_info_v0_0']['filename'])
            filenames_v0_1 = osp.join(results['img_prefix'], results['img_info_v0_1']['filename'])
            filenames_v1_0 = osp.join(results['img_prefix'], results['img_info_v1_0']['filename'])
            filenames_v1_1 = osp.join(results['img_prefix'], results['img_info_v1_1']['filename'])
        else:
            filenames_v0_0 = results['img_info_v0_0']['filename']
            filenames_v0_1 = results['img_info_v0_1']['filename']
            filenames_v1_0 = results['img_info_v1_0']['filename']
            filenames_v1_1 = results['img_info_v1_1']['filename']

        img_bytes = self.file_client.get(filenames_v0_0)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)
        results['img_v0_0'] = img

        img_bytes = self.file_client.get(filenames_v0_1)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)
        results['img_v0_1'] = img

        img_bytes = self.file_client.get(filenames_v1_0)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)
        results['img_v1_0'] = img

        img_bytes = self.file_client.get(filenames_v1_1)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)
        results['img_v1_1'] = img

        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32), std=np.ones(num_channels, dtype=np.float32), to_rgb=False
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations_Semi(object):
    def __init__(self, reduce_zero_label=False, file_client_args=dict(backend='disk'), imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'], results['img_info_v0_0']['ann'])
        else:
            filename = results['img_info_v0_0']['ann']

        img_bytes = self.file_client.get(filename)
        gt = mmcv.imfrombytes(img_bytes, flag='unchanged', backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt[gt == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt[gt == 0] = 255
            gt = gt - 1
            gt[gt == 254] = 255
        results['gt'] = gt
        results['seg_fields'].append('gt')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
