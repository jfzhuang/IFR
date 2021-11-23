import torch
import torch.nn as nn

from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmseg.models.backbones.resnet_backbone import ResNetBackbone

from mmseg.utils import get_root_logger

from ..builder import BACKBONES
from ..utils import ResLayer


@BACKBONES.register_module()
class ResNet18(nn.Module):
    def __init__(self, norm_type='sync_batchnorm'):
        super(ResNet18, self).__init__()
        pretrained = './pretrained/resnet18-imagenet.pth'
        model = ResNetBackbone(backbone='deepbase_resnet18_dilated8', pretrained=pretrained, norm_type=norm_type)
        self.stem = nn.Sequential(model.prefix, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        """Forward function."""
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return tuple([x])

    def train(self, mode=True):
        super(ResNet18, self).train(mode)


@BACKBONES.register_module()
class ResNet50(nn.Module):
    def __init__(self, norm_type='sync_batchnorm'):
        super(ResNet50, self).__init__()
        pretrained = './pretrained/resnet50-imagenet.pth'
        model = ResNetBackbone(backbone='deepbase_resnet50_dilated8', pretrained=pretrained, norm_type=norm_type)
        self.stem = nn.Sequential(model.prefix, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        """Forward function."""
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return tuple([x])

    def train(self, mode=True):
        super(ResNet50, self).train(mode)


@BACKBONES.register_module()
class ResNet101(nn.Module):
    def __init__(self, norm_type='sync_batchnorm'):
        super(ResNet101, self).__init__()
        pretrained = './pretrained/resnet101-imagenet.pth'
        model = ResNetBackbone(backbone='deepbase_resnet101_dilated8', pretrained=pretrained, norm_type=norm_type)
        self.stem = nn.Sequential(model.prefix, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        """Forward function."""
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return tuple([x])

    def train(self, mode=True):
        super(ResNet101, self).train(mode)


if __name__ == '__main__':
    model = ResNet101()
    model.cuda().eval()

    data = torch.rand([1, 3, 512, 1024]).cuda()
    with torch.no_grad():
        outs = model(data)
        for i in range(len(outs)):
            print(outs[i].shape)
