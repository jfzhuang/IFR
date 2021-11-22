import torch
import torch.nn as nn
from torch.nn import init


class FlowNets(nn.Module):
    def __init__(self, in_planes=6, batchNorm=False):

        super(FlowNets, self).__init__()

        self.conv1 = self.conv(in_planes, 24, batchNorm, kernel_size=7, stride=2, padding=3)
        self.conv2 = self.conv(24, 48, batchNorm, kernel_size=5, stride=2, padding=2)
        self.conv3 = self.conv(48, 96, batchNorm, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = self.conv(96, 96, batchNorm, kernel_size=3, stride=1, padding=1)
        self.conv4 = self.conv(96, 192, batchNorm, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = self.conv(192, 192, batchNorm, kernel_size=3, stride=1, padding=1)
        self.conv5 = self.conv(192, 192, batchNorm, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = self.conv(192, 192, batchNorm, kernel_size=3, stride=1, padding=1)
        self.conv6 = self.conv(192, 384, batchNorm, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = self.conv(384, 384, batchNorm, kernel_size=3, stride=1, padding=1)

        self.predict_flow6 = self.predict_flow(384)
        self.predict_flow5 = self.predict_flow(386)
        self.predict_flow4 = self.predict_flow(290)
        self.predict_flow3 = self.predict_flow(146)
        self.predict_flow2 = self.predict_flow(74)

        self.deconv5 = self.deconv(384, 192)
        self.deconv4 = self.deconv(386, 96)
        self.deconv3 = self.deconv(290, 48)
        self.deconv2 = self.deconv(146, 24)

        self.upsample_flow6to5 = nn.ConvTranspose2d(2, 2, 4, 2, 0, bias=True)
        self.upsample_flow5to4 = nn.ConvTranspose2d(2, 2, 4, 2, 0, bias=True)
        self.upsample_flow4to3 = nn.ConvTranspose2d(2, 2, 4, 2, 0, bias=True)
        self.upsample_flow3to2 = nn.ConvTranspose2d(2, 2, 4, 2, 0, bias=True)

    def forward(self, x):
        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.antipad(
            self.upsample_flow6to5(flow6), evenh=out_conv5.shape[2] % 2 == 0, evenw=out_conv5.shape[3] % 2 == 0
        )
        out_deconv5 = self.antipad(
            self.deconv5(out_conv6), evenh=out_conv5.shape[2] % 2 == 0, evenw=out_conv5.shape[3] % 2 == 0
        )

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.antipad(
            self.upsample_flow5to4(flow5), evenh=out_conv4.shape[2] % 2 == 0, evenw=out_conv4.shape[3] % 2 == 0
        )
        out_deconv4 = self.antipad(
            self.deconv4(concat5), evenh=out_conv4.shape[2] % 2 == 0, evenw=out_conv4.shape[3] % 2 == 0
        )

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.antipad(
            self.upsample_flow4to3(flow4), evenh=out_conv3.shape[2] % 2 == 0, evenw=out_conv3.shape[3] % 2 == 0
        )
        out_deconv3 = self.antipad(
            self.deconv3(concat4), evenh=out_conv3.shape[2] % 2 == 0, evenw=out_conv3.shape[3] % 2 == 0
        )

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.antipad(
            self.upsample_flow3to2(flow3), evenh=out_conv2.shape[2] % 2 == 0, evenw=out_conv2.shape[3] % 2 == 0
        )
        out_deconv2 = self.antipad(
            self.deconv2(concat3), evenh=out_conv2.shape[2] % 2 == 0, evenw=out_conv2.shape[3] % 2 == 0
        )

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        return flow2 * 5.0

    def conv(self, in_planes, out_planes, batchNorm, kernel_size=3, stride=1, padding=1):
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.SyncBatchNorm(out_planes),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
            )

    def predict_flow(self, in_planes):
        return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)

    def deconv(self, in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def antipad(self, tensor, evenh=True, evenw=True, num=1):

        h = tensor.shape[2]
        w = tensor.shape[3]
        if evenh and evenw:
            tensor = tensor.narrow(2, 1, h - 2 * num)
            tensor = tensor.narrow(3, 1, w - 2 * num)
            return tensor
        elif evenh and (not evenw):
            tensor = tensor.narrow(2, 1, h - 2 * num)
            tensor = tensor.narrow(3, 1, w - 2 * num - 1)
            return tensor
        elif (not evenh) and evenw:
            tensor = tensor.narrow(2, 1, h - 2 * num - 1)
            tensor = tensor.narrow(3, 1, w - 2 * num)
            return tensor
        else:
            tensor = tensor.narrow(2, 1, h - 2 * num - 1)
            tensor = tensor.narrow(3, 1, w - 2 * num - 1)
            return tensor


if __name__ == "__main__":
    import time

    model = FlowNets()
    model.cuda().eval()
    image = torch.randn(1, 6, 512, 1024).cuda()

    t1 = time.time()
    with torch.no_grad():
        for i in range(100):
            out, _ = model.forward(image)
            print(i, out.shape)
    t2 = time.time()
    print('cost:{}'.format((t2 - t1) / 100))
