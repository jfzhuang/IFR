import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def gen_propotype(feat, seg, unique):
    prototypes = []
    for i in range(len(unique)):
        mask = (seg == unique[i]).float()
        prototype = (feat * mask).sum(dim=-1) / mask.sum(dim=-1)
        prototypes.append(prototype.unsqueeze(0))
    return prototypes


def gen_boundary_func(mask, block_size, boundary_width):
    b = block_size
    bw = boundary_width
    h, w = mask.shape

    tmp = torch.zeros([h, w, b * b]).long()
    for i in range(bw):
        for j in range(bw):
            mask_patch = mask[i:i + h - b, j:j + w - b]
            tmp[b // 2:h - b//2, b // 2:w - b//2, i*bw + j] = mask_patch
    tmp_min = tmp.min(dim=2)[0].float()
    tmp_max = tmp.max(dim=2)[0].float()
    boundary = (tmp_min != tmp_max).float()
    return boundary


def gen_boundary(masks, block_size, boundary_width):
    _, h, w = masks[0].shape

    boundary = 1.0
    for i in range(len(masks)):
        boundary_ = gen_boundary_func(masks[i].squeeze(), block_size, boundary_width)
        boundary *= boundary_

    if boundary.sum() == 0:
        return None
    return boundary


@LOSSES.register_module()
class TemporalLoss(nn.Module):
    def __init__(self,
                 block_size,
                 regular=False,
                 num_proposal=32,
                 norm=True,
                 boundary_width=2,
                 loss_weight=1.0,
                 second_order=True):
        super(TemporalLoss, self).__init__()
        self.block_size = block_size
        self.regular = regular
        self.num_proposal = num_proposal
        self.norm = norm
        self.boundary_width = boundary_width
        self.loss_weight = loss_weight
        self.second_order = second_order
        self.loss_func = nn.L1Loss()

    def forward(self, inputs):
        """Forward function."""
        feats, masks = inputs

        num_frame = len(feats)
        for i in range(num_frame):
            assert feats[i].shape[2:] == masks[i].shape[2:]
        if self.norm:
            for i in range(num_frame):
                feats[i] = F.normalize(feats[i], dim=1, p=2)

        n, c, h, w = feats[0].shape
        b = self.block_size
        if self.boundary_width is None:
            boundary_width = b

        proposals_batch = []
        if self.regular:
            for _ in range(n):
                proposals = []
                for i in range(h // b):
                    for j in range(w // b):
                        proposals.append([i * b, j * b])
                proposals_batch.append(proposals)
        else:
            for i in range(n):
                proposals = []
                masks_sample = []
                for nf in range(num_frame):
                    masks_sample.append(masks[nf][i, ...])
                boundary_map = gen_boundary(masks_sample, self.block_size, self.boundary_width)
                if boundary_map is None:
                    proposals_batch.append(proposals)
                    continue

                boundary_map = boundary_map.view(-1)
                if boundary_map.sum().item() < self.num_proposal:
                    idx_list = torch.multinomial(boundary_map, self.num_proposal, replacement=True)
                else:
                    idx_list = torch.multinomial(boundary_map, self.num_proposal, replacement=False)
                for idx in idx_list:
                    idx_h = idx//w - b//2
                    idx_w = idx%w - b//2
                    proposals.append([idx_h, idx_w])
                proposals_batch.append(proposals)

        loss = 0.0
        count = 0
        for i in range(n):
            proposals = proposals_batch[i]
            for p in proposals:
                p_h, p_w = p
                sub_masks = []
                unique_list_tmp = []
                for n_f in range(num_frame):
                    sub_m = masks[n_f][i, :, p_h:p_h + b, p_w:p_w + b].contiguous()
                    sub_m = sub_m.view(1, b * b)
                    unique = torch.unique(sub_m)
                    sub_masks.append(sub_m)
                    unique_list_tmp.append(unique)

                unique_list = []
                tmp = {}
                for unique in unique_list_tmp:
                    for unique_value in unique:
                        unique_value = unique_value.long().item()
                        if unique_value not in tmp.keys():
                            tmp[unique_value] = 1
                        else:
                            tmp[unique_value] += 1
                for k, v in tmp.items():
                    if v == num_frame:
                        unique_list.append(k)
                if len(unique_list) == 0:
                    continue

                prototype_list = []
                for n_f in range(num_frame):
                    sub_input = feats[n_f][i, :, p_h:p_h + b, p_w:p_w + b].contiguous()
                    sub_input = sub_input.view(c, b * b)
                    prototype = gen_propotype(sub_input, sub_masks[n_f], unique_list)
                    prototype_list.append(prototype)

                num_class = len(prototype_list[0])
                loss_sample = 0.0
                for n_c in range(num_class):
                    for n_f in range(num_frame - 2):
                        if self.second_order:
                            tmp = self.loss_func(prototype_list[n_f][n_c] - prototype_list[n_f + 1][n_c],
                                                 prototype_list[n_f + 1][n_c] - prototype_list[n_f + 2][n_c])
                        else:
                            tmp = (self.loss_func(prototype_list[n_f][n_c], prototype_list[n_f + 1][n_c]) +
                                   self.loss_func(prototype_list[n_f + 1][n_c], prototype_list[n_f + 2][n_c]))
                        loss_sample += tmp
                loss_sample /= (num_class * (num_frame-2))
                loss += loss_sample
                count += 1

        if count > 0:
            loss /= count
        loss = loss * self.loss_weight
        return loss


@LOSSES.register_module()
class TemporalLoss_Naive(nn.Module):
    def __init__(self, down_scale=1.0, up_scale=1.0, norm=True, second_order=False, loss_weight=1.0, power=1.0):
        super(TemporalLoss_Naive, self).__init__()
        self.loss_weight = loss_weight
        self.down_scale = down_scale
        self.up_scale = up_scale
        self.second_order = second_order
        self.norm = norm
        self.power = power
        self.loss_func = nn.L1Loss()

    def forward(self, inputs):
        """Forward function."""
        feats, _ = inputs
        if self.up_scale > 1:
            for i in range(len(feats)):
                feats[i] = F.interpolate(feats[i], scale_factor=self.up_scale, mode='bilinear')

        if self.down_scale > 1:
            for i in range(len(feats)):
                feats[i] = F.interpolate(feats[i], scale_factor=1 / self.down_scale, mode='bilinear')

        if self.norm:
            for i in range(len(feats)):
                feats[i] = F.normalize(feats[i], dim=1, p=2)

        loss = 0.0
        if self.second_order:
            for i in range(len(feats) - 2):
                loss += self.loss_func(feats[i] - feats[i + 1], feats[i + 1] - feats[i + 2])
            loss /= (len(feats) - 2)
        else:
            for i in range(len(feats) - 1):
                loss += self.loss_func(feats[i], feats[i + 1])
            loss /= (len(feats) - 1)

        loss = loss * self.loss_weight
        return loss


@LOSSES.register_module()
class TemporalLoss_NoClassWise(nn.Module):
    def __init__(self,
                 block_size,
                 regular=False,
                 num_proposal=32,
                 norm=True,
                 boundary=False,
                 boundary_width=2,
                 mean=False,
                 loss_weight=1.0,
                 class_weight=None,
                 second_order=False,
                 both=False):
        super(TemporalLoss_NoClassWise, self).__init__()
        self.block_size = block_size
        self.regular = regular
        self.num_proposal = num_proposal
        self.norm = norm
        self.boundary = boundary
        self.boundary_width = boundary_width
        self.mean = mean
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.second_order = second_order
        self.both = both
        self.loss_func = nn.L1Loss()

    def forward(self, inputs):
        """Forward function."""
        feats, masks = inputs

        num_frame = len(feats)
        for i in range(num_frame):
            assert feats[i].shape[2:] == masks[i].shape[2:]
        if self.norm:
            for i in range(num_frame):
                feats[i] = F.normalize(feats[i], dim=1, p=2)

        n, c, h, w = feats[0].shape
        b = self.block_size
        if self.boundary_width is None:
            boundary_width = b

        proposals_batch = []
        if self.regular:
            for _ in range(n):
                proposals = []
                for i in range(h // b):
                    for j in range(w // b):
                        proposals.append([i * b, j * b])
                proposals_batch.append(proposals)
        else:
            for i in range(n):
                proposals = []
                masks_sample = []
                for nf in range(num_frame):
                    masks_sample.append(masks[nf][i, ...])
                boundary_map = gen_boundary(masks_sample, self.block_size, self.boundary_width)
                if boundary_map is None:
                    for _ in range(n):
                        proposals = []
                        for i in range(h // b):
                            for j in range(w // b):
                                proposals.append([i * b, j * b])
                        proposals_batch.append(proposals)
                    continue

                boundary_map = boundary_map.view(-1)
                if boundary_map.sum().item() < self.num_proposal:
                    idx_list = torch.multinomial(boundary_map, self.num_proposal, replacement=True)
                else:
                    idx_list = torch.multinomial(boundary_map, self.num_proposal, replacement=False)
                for idx in idx_list:
                    idx_h = idx//w - b//2
                    idx_w = idx%w - b//2
                    proposals.append([idx_h, idx_w])
                proposals_batch.append(proposals)

        loss = 0.0
        count = 0
        for i in range(n):
            proposals = proposals_batch[i]
            for p in proposals:
                p_h, p_w = p

                feat_list = []
                for n_f in range(num_frame):
                    feat = feats[n_f][i, :, p_h:p_h + b, p_w:p_w + b]
                    if self.mean:
                        feat = feat.mean(1).mean(1)
                    feat_list.append(feat)

                loss_sample = 0.0
                if self.second_order:
                    for n_f in range(num_frame - 2):
                        loss_sample += self.loss_func(feat_list[n_f] - feat_list[n_f + 1],
                                                      feat_list[n_f + 1] - feat_list[n_f + 2])
                    loss_sample /= (num_frame - 2)
                else:
                    for n_f in range(num_frame - 1):
                        loss_sample += self.loss_func(feat_list[n_f], feat_list[n_f + 1])
                    loss_sample /= (num_frame - 1)
                if self.both:
                    loss_sample_L1 = 0.0
                    for n_f in range(num_frame - 1):
                        loss_sample_L1 += self.loss_func(feat_list[n_f], feat_list[n_f + 1])
                    loss_sample_L1 /= (num_frame - 1)
                    loss_sample += loss_sample_L1
                loss += loss_sample
                count += 1

        if count > 0:
            loss /= count
        loss = loss * self.loss_weight
        return loss


@LOSSES.register_module()
class TemporalLoss_NoClassWise_Directional(nn.Module):
    def __init__(self,
                 block_size,
                 regular=False,
                 num_proposal=32,
                 norm=True,
                 boundary_width=2,
                 loss_weight=1.0,
                 second_order=False):
        super(TemporalLoss_NoClassWise_Directional, self).__init__()
        self.block_size = block_size
        self.regular = regular
        self.num_proposal = num_proposal
        self.norm = norm
        self.boundary_width = boundary_width
        self.loss_weight = loss_weight
        self.second_order = second_order
        self.loss_func = nn.L1Loss()

    def forward(self, inputs):
        """Forward function."""
        feats, scores, masks = inputs

        num_frame = len(feats)
        for i in range(num_frame):
            assert feats[i].shape[2:] == masks[i].shape[2:]
        if self.norm:
            for i in range(num_frame):
                feats[i] = F.normalize(feats[i], dim=1, p=2)

        scores = torch.cat(scores, dim=1)
        directional_masks = torch.argmax(scores, dim=1, keepdims=True)
        for i in range(len(feats)):
            directional_mask = (directional_masks == i).float()
            feat = feats[i]
            feat_detach = feat.detach()
            feats[i] = feat * (1-directional_mask) + feat_detach*directional_mask

        n, c, h, w = feats[0].shape
        b = self.block_size
        if self.boundary_width is None:
            boundary_width = b

        proposals_batch = []
        if self.regular:
            for _ in range(n):
                proposals = []
                for i in range(h // b):
                    for j in range(w // b):
                        proposals.append([i * b, j * b])
                proposals_batch.append(proposals)
        else:
            for i in range(n):
                proposals = []
                masks_sample = []
                for nf in range(num_frame):
                    masks_sample.append(masks[nf][i, ...])
                boundary_map = gen_boundary(masks_sample, self.block_size, self.boundary_width)
                if boundary_map is None:
                    proposals_batch.append(proposals)
                    continue

                boundary_map = boundary_map.view(-1)
                if boundary_map.sum().item() < self.num_proposal:
                    idx_list = torch.multinomial(boundary_map, self.num_proposal, replacement=True)
                else:
                    idx_list = torch.multinomial(boundary_map, self.num_proposal, replacement=False)
                for idx in idx_list:
                    idx_h = idx//w - b//2
                    idx_w = idx%w - b//2
                    proposals.append([idx_h, idx_w])
                proposals_batch.append(proposals)

        loss = 0.0
        count = 0
        for i in range(n):
            proposals = proposals_batch[i]
            for p in proposals:
                p_h, p_w = p

                feat_list = []
                for n_f in range(num_frame):
                    feat = feats[n_f][i, :, p_h:p_h + b, p_w:p_w + b]
                    feat_list.append(feat)

                loss_sample = 0.0
                if self.second_order:
                    for n_f in range(num_frame - 2):
                        loss_sample += self.loss_func(feat_list[n_f] - feat_list[n_f + 1],
                                                      feat_list[n_f + 1] - feat_list[n_f + 2])
                    loss_sample /= (num_frame - 2)
                else:
                    for n_f in range(num_frame - 1):
                        loss_sample += self.loss_func(feat_list[n_f], feat_list[n_f + 1])
                    loss_sample /= (num_frame - 1)
                loss += loss_sample
                count += 1

        if count > 0:
            loss /= count
        loss = loss * self.loss_weight
        return loss