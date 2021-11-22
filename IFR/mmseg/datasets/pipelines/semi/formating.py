from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from mmseg.datasets.builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class DefaultFormatBundle_Semi(object):
    def __call__(self, results):
        if 'gt' in results:
            results['gt'] = DC(to_tensor(results['gt'][None, ...].astype(np.int64)), stack=True)
        if 'img_v0_0' in results:
            img = results['img_v0_0']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img_v0_0'] = DC(to_tensor(img), stack=True)
        if 'img_v0_1' in results:
            img = results['img_v0_1']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img_v0_1'] = DC(to_tensor(img), stack=True)
        if 'img_v0_1_s' in results:
            img = results['img_v0_1_s']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img_v0_1_s'] = DC(to_tensor(img), stack=True)
        if 'img_v1_0' in results:
            img = results['img_v1_0']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img_v1_0'] = DC(to_tensor(img), stack=True)
        if 'img_v1_1' in results:
            img = results['img_v1_1']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img_v1_1'] = DC(to_tensor(img), stack=True)
        if 'img_v1_1_s' in results:
            img = results['img_v1_1_s']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img_v1_1_s'] = DC(to_tensor(img), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__
