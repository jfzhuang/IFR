import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models import build_segmentor
from mmcv.utils import Config

from lib.model.pspnet import pspnet_res18, pspnet_res101
from lib.model.flownet import FlowNets
from lib.model.warpnet import warp


class Accel18(nn.Module):
    def __init__(self, num_classes=19, weight_res18=None, weight_res101=None, weight_flownet=None):
        super(Accel18, self).__init__()
        self.net_ref = pspnet_res101()
        self.net_update = pspnet_res18()
        self.merge = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1, padding=0)

        self.flownet = FlowNets()
        self.warp = warp()

        self.weight_init(weight_res18, weight_res101, weight_flownet)

        self.criterion_semantic = nn.CrossEntropyLoss(ignore_index=255)

    def weight_init(self, weight_res18, weight_res101, weight_flownet):
        if weight_res18 is not None:
            weight = torch.load(weight_res18, map_location='cpu')
            weight = weight['state_dict']
            self.net_update.load_state_dict(weight, True)
            self.net_update.fix_backbone()

        if weight_res101 is not None:
            weight = torch.load(weight_res101, map_location='cpu')
            weight = weight['state_dict']
            self.net_ref.load_state_dict(weight, True)
            self.net_ref.fix_backbone()

        if weight_flownet is not None:
            weight = torch.load(weight_flownet, map_location='cpu')
            self.flownet.load_state_dict(weight, True)

        nn.init.xavier_normal_(self.merge.weight)
        self.merge.bias.data.fill_(0)
        print('pretrained weight loaded')

    def forward(self, im_seg_list, im_flow_list, gt=None):
        n, c, t, h, w = im_seg_list.shape
        pred = self.net_ref(im_seg_list[:, :, 0, :, :])
        pred = F.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=False)
        for i in range(t - 1):
            flow = self.flownet(torch.cat([im_flow_list[:, :, i + 1, :, :], im_flow_list[:, :, i, :, :]], dim=1))
            pred = self.warp(pred, flow)
        pred_update = self.net_update(im_seg_list[:, :, -1, :, :])
        pred_update = F.interpolate(pred_update, scale_factor=2, mode='bilinear', align_corners=False)
        pred_merge = self.merge(torch.cat([pred, pred_update], dim=1))
        pred_merge = F.interpolate(pred_merge, scale_factor=4, mode='bilinear', align_corners=False)
        if gt is not None:
            loss = self.criterion_semantic(pred_merge, gt)
            loss = loss.unsqueeze(0)
            return loss
        else:
            return pred_merge

    def evaluate(self, im_seg_list, im_flow_list):
        out_list = []
        t = im_seg_list.shape[2]
        pred = self.net_ref(im_seg_list[:, :, 0, :, :])
        pred = F.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(pred, scale_factor=4, mode='bilinear', align_corners=False)
        out = torch.argmax(out, dim=1)
        out_list.append(out)

        for i in range(t - 1):
            flow = self.flownet(torch.cat([im_flow_list[:, :, i + 1, :, :], im_flow_list[:, :, i, :, :]], dim=1))
            pred = self.warp(pred, flow)

            pred_update = self.net_update(im_seg_list[:, :, -1, :, :])
            pred_update = F.interpolate(pred_update, scale_factor=2, mode='bilinear', align_corners=False)
            pred_merge = self.merge(torch.cat([pred, pred_update], dim=1))
            pred_merge = F.interpolate(pred_merge, scale_factor=4, mode='bilinear', align_corners=False)
            out = torch.argmax(pred_merge, dim=1)
            out_list.append(out)

        return out_list

    def set_train(self):
        self.net_ref.eval()
        self.net_ref.model.decode_head.conv_seg.train()
        self.net_update.eval()
        self.net_update.model.decode_head.conv_seg.train()
        self.flownet.train()
        self.merge.train()


if __name__ == '__main__':
    model = Accel18(weight_res18=None, weight_res101=None, weight_flownet=None)
    model.cuda().eval()

    im_seg_list = torch.rand([1, 3, 5, 512, 1024]).cuda()
    im_flow_list = torch.rand([1, 3, 5, 512, 1024]).cuda()
    with torch.no_grad():
        out_list = model.evaluate(im_seg_list, im_flow_list)
        print(len(out_list), out_list[0].shape)
