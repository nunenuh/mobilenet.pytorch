import torch
import torch.nn as nn


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, stride=1, padding=1, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ksize,
                              stride=stride, padding=padding, group=in_chan, bias=bias)

    def forward(self, input):
        return self.conv(input)

class ConvBnRelu(nn.Module):
    def __init__(self, nin, nout, ksize=3, strd=1, pad=1, bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=ksize, stride=strd, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MobileNetV1Block(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(MobileNetV1Block, self).__init__()
        self.depthwise_conv3x3 = DepthwiseConv2d(in_chan, in_chan)
        self.bn1 = nn.BatchNorm2d(in_chan)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.depthwise_conv3x3(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1x1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()

