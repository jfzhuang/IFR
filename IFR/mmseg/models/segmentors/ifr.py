import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor, SemiBaseSegmentor
from ..losses import accuracy
from torch.distributions.uniform import Uniform


@SEGMENTORS.register_module()
class IFR(SemiBaseSegmentor):
    def __init__(
        self,
        backbone,
        decode_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        weight_re_labeled=1.0,
        weight_re_unlabeled=1.0,
        weight_re_strong=1.0,
        temperature=1.0,
    ):
        super(IFR, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self._init_decode_head(decode_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.temperature = temperature
        self.temperature_logits = 0.01
        self.rampup_length = train_cfg['rampup_length']
        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

        self.weight_re_labeled = weight_re_labeled
        self.weight_re_unlabeled = weight_re_unlabeled
        self.weight_re_strong = weight_re_strong

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def init_weights(self, pretrained=None):
        super(IFR, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()

    def encode_decode(self, img, img_metas=None):
        x = self.backbone(img)
        out = self.decode_head(x)
        out = resize(input=out, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
        return out

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def forward(self, return_loss=True, img_metas=None, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(img_metas=img_metas, **kwargs)

    def gen_prototypes(self, feat, logit):
        n, c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous().view(n, h * w, -1)
        logit = logit.permute(0, 2, 3, 1).contiguous().view(n, h * w, -1)
        label = torch.argmax(logit, dim=-1)
        prototypes_batch = []
        for i in range(n):
            classes = torch.unique(label[i, :].clone().detach())
            prototypes = []
            for c in classes:
                prototype = feat[label == c, :]
                prototype = prototype.mean(0, keepdims=True)
                prototypes.append(prototype)
            prototypes = torch.cat(prototypes, dim=0)
            prototypes = prototypes.permute(1, 0).contiguous()
            prototypes_batch.append(prototypes)
        return prototypes_batch

    def reconstruct(self, feat, prototypes):
        c, h, w = feat.shape
        feat = feat.view(c, h * w).permute(1, 0).contiguous()
        feat_norm = F.normalize(feat, dim=-1)
        prototypes_norm = F.normalize(prototypes, dim=0)
        attn = torch.mm(feat_norm, prototypes_norm) / self.temperature
        attn = torch.softmax(attn, dim=-1)
        prototypes = prototypes.permute(1, 0).contiguous()
        feat_re = torch.mm(attn, prototypes)
        feat_re = feat_re.view(h, w, c).permute(2, 0, 1).contiguous().unsqueeze(0)
        return feat_re

    def forward_train(self, img_v0_0, img_v0_1, img_v0_1_s, img_v1_0, img_v1_1, img_v1_1_s, gt, iter):
        n, c, h, w = img_v0_1.shape
        gt = gt.squeeze(1)

        losses = dict()

        # supervised loss
        feats_v0_0 = self.backbone(img_v0_0)
        feats_v0_0 = self.decode_head(feats_v0_0, return_feat=True)
        logits_v0_0 = self.decode_head.cls_seg(feats_v0_0)
        logits_v0_0 = F.interpolate(logits_v0_0, size=gt.shape[1:], mode='bilinear', align_corners=self.align_corners)
        loss = dict()
        loss['loss_seg'] = self.decode_head.loss_decode(logits_v0_0, gt, ignore_index=255)
        loss['acc_seg'] = accuracy(logits_v0_0, gt)
        losses.update(add_prefix(loss, 'decode'))

        n = feats_v0_0.shape[0]
        loss_seg_unsup = 0.0

        # IFR on labeled videos
        feats_v0_1 = self.backbone(img_v0_1)
        feats_v0_1 = self.decode_head(feats_v0_1, return_feat=True)
        logits_v0_1 = self.decode_head.cls_seg(feats_v0_1)

        prototypes_batch = self.gen_prototypes(feats_v0_1, logits_v0_1)
        feat_re_batch = []
        for j in range(n):
            feat_re = self.reconstruct(feats_v0_0[j, ...], prototypes_batch[j])
            feat_re_batch.append(feat_re)
        feat_re_batch = torch.cat(feat_re_batch, dim=0)

        logits_sup_re = self.decode_head.cls_seg(feat_re_batch)
        logits_sup_re = F.interpolate(
            logits_sup_re, size=gt.shape[1:], mode='bilinear', align_corners=self.align_corners
        )
        loss_seg_unsup += (
            self.decode_head.loss_decode(logits_sup_re / self.temperature_logits, gt, ignore_index=255)
            * self.weight_re_labeled
        )

        # IFR on labeled augmented videos
        feats_v0_1_s = self.backbone(img_v0_1_s)
        feats_v0_1_s = self.decode_head(feats_v0_1_s, return_feat=True)

        prototypes_batch = self.gen_prototypes(feats_v0_1_s, logits_v0_1)
        feat_re_batch = []
        for j in range(n):
            feat_re = self.reconstruct(feats_v0_0[j, ...], prototypes_batch[j])
            feat_re_batch.append(feat_re)
        feat_re_batch = torch.cat(feat_re_batch, dim=0)

        logits_sup_re = self.decode_head.cls_seg(feat_re_batch)
        logits_sup_re = F.interpolate(
            logits_sup_re, size=gt.shape[1:], mode='bilinear', align_corners=self.align_corners
        )
        loss_seg_unsup += (
            self.decode_head.loss_decode(logits_sup_re / self.temperature_logits, gt, ignore_index=255)
            * self.weight_re_labeled
            * self.weight_re_strong
        )

        # IFR on unlabeled videos
        feats_v1_0 = self.backbone(img_v1_0)
        feats_v1_0 = self.decode_head(feats_v1_0, return_feat=True)
        logits_v1_0 = self.decode_head.cls_seg(feats_v1_0)
        logits_v1_0 = F.interpolate(logits_v1_0, size=gt.shape[1:], mode='bilinear', align_corners=self.align_corners)
        pseudo_label = torch.argmax(logits_v1_0, dim=1)

        feats_v1_1 = self.backbone(img_v1_1)
        feats_v1_1 = self.decode_head(feats_v1_1, return_feat=True)
        logits_v1_1 = self.decode_head.cls_seg(feats_v1_1)

        prototypes_batch = self.gen_prototypes(feats_v1_1, logits_v1_1)
        feat_re_batch = []
        for j in range(n):
            feat_re = self.reconstruct(feats_v1_0[j, ...], prototypes_batch[j])
            feat_re_batch.append(feat_re)
        feat_re_batch = torch.cat(feat_re_batch, dim=0)

        logits_sup_re = self.decode_head.cls_seg(feat_re_batch)
        logits_sup_re = F.interpolate(
            logits_sup_re, size=gt.shape[1:], mode='bilinear', align_corners=self.align_corners
        )
        loss_seg_unsup += (
            self.decode_head.loss_decode(logits_sup_re / self.temperature_logits, pseudo_label, ignore_index=255)
            * self.weight_re_unlabeled
        )

        # IFR on unlabeled augmented videos
        feats_v1_1_s = self.backbone(img_v1_1_s)
        feats_v1_1_s = self.decode_head(feats_v1_1_s, return_feat=True)

        prototypes_batch = self.gen_prototypes(feats_v1_1_s, logits_v1_1)
        feat_re_batch = []
        for j in range(n):
            feat_re = self.reconstruct(feats_v1_0[j, ...], prototypes_batch[j])
            feat_re_batch.append(feat_re)
        feat_re_batch = torch.cat(feat_re_batch, dim=0)

        logits_sup_re = self.decode_head.cls_seg(feat_re_batch)
        logits_sup_re = F.interpolate(
            logits_sup_re, size=gt.shape[1:], mode='bilinear', align_corners=self.align_corners
        )
        loss_seg_unsup += (
            self.decode_head.loss_decode(logits_sup_re / self.temperature_logits, pseudo_label, ignore_index=255)
            * self.weight_re_unlabeled
            * self.weight_re_strong
        )

        weight_unsup = self.sigmoid_rampup(iter, self.rampup_length)
        loss['loss_seg_unsup'] = loss_seg_unsup * weight_unsup
        losses.update(add_prefix(loss, 'decode'))

        weight_unsup = torch.Tensor([weight_unsup]).to(gt.device)
        losses['rampup'] = weight_unsup

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False,
            )
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(seg_logit, size=size, mode='bilinear', align_corners=self.align_corners, warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
