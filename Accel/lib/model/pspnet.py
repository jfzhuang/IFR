import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models import build_segmentor
from mmcv.utils import Config


class pspnet_res18(nn.Module):
    def __init__(self, num_classes=19):
        super(pspnet_res18, self).__init__()
        config_path = './IFR/configs/_base_/models/pspnet_r18-d8.py'
        cfg = Config.fromfile(config_path)
        self.model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        self.fix_backbone()

    def fix_backbone(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.model.decode_head.conv_seg.parameters():
            p.requires_grad = True

    def forward(self, x):
        with torch.no_grad():
            feat = self.model.backbone(x)
            feat = self.model.decode_head(feat, return_feat=True)
        pred = self.model.decode_head.cls_seg(feat)
        return pred


class pspnet_res101(nn.Module):
    def __init__(self, num_classes=19):
        super(pspnet_res101, self).__init__()
        config_path = './IFR/configs/_base_/models/pspnet_r101-d8.py'
        cfg = Config.fromfile(config_path)
        self.model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        self.fix_backbone()

    def fix_backbone(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.model.decode_head.conv_seg.parameters():
            p.requires_grad = True

    def forward(self, x):
        with torch.no_grad():
            feat = self.model.backbone(x)
            feat = self.model.decode_head(feat, return_feat=True)
        pred = self.model.decode_head.cls_seg(feat)
        return pred


if __name__ == "__main__":
    model = pspnet_res101()
    model.cuda().eval()

    im = torch.rand(1, 3, 512, 1024).cuda()
    with torch.no_grad():
        out = model(im)
        print(out.shape)
