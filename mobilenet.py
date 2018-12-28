import torch
import torch.nn as nn


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=in_chan, bias=bias)

    def forward(self, input):
        return self.conv(input)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class Conv3x3BnRelu(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, strd=1, pad=1, bias=False):
        super(Conv3x3BnRelu, self).__init__()
        self.conv3x3 = nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=strd, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv3x3(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWConv3x3BnRelu(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, strd=1, pad=1, bias=False):
        super(DWConv3x3BnRelu, self).__init__()
        self.dwconv3x3 = DepthwiseConv2d(in_chan, out_chan, kernel_size=ksize, stride=strd, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.dwconv3x3(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1BnRelu(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=1, strd=1, pad=0, bias=False):
        super(Conv1x1BnRelu, self).__init__()
        self.conv1x1 = nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=strd, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1x1(input)
        x = self.bn(x)
        x = self.relu(x)
        return x




class ReadableMobileNetV1(nn.Module):
    def __init__(self, in_chan=3, classes=1000):
        super(ReadableMobileNetV1, self).__init__()
        self.c3br1 = Conv3x3BnRelu(in_chan, 32, strd=2, pad=1)

        self.dwc3br1 = DWConv3x3BnRelu(32, 32)
        self.c1br1 = Conv1x1BnRelu(32, 64)

        self.dwc3br2 = DWConv3x3BnRelu(64, 64, strd=2)
        self.c1br2 = Conv1x1BnRelu(64, 128)

        self.dwc3br3 = DWConv3x3BnRelu(128, 128)
        self.c1br3 = Conv1x1BnRelu(128, 128)

        self.dwc3br4 = DWConv3x3BnRelu(128, 128, strd=2)
        self.c1br4 = Conv1x1BnRelu(128, 256)

        self.dwc3br5 = DWConv3x3BnRelu(256, 256)
        self.c1br5 = Conv1x1BnRelu(256, 256)

        self.dwc3br6 = DWConv3x3BnRelu(256, 256, strd=2)
        self.c1br6 = Conv1x1BnRelu(256, 512)

        self.middle5x = nn.ModuleList()
        for i in range(5):
            self.middle5x.append(DWConv3x3BnRelu(512, 512))
            self.middle5x.append(Conv1x1BnRelu(512, 512))

        self.dwc3br7 = DWConv3x3BnRelu(512, 512, strd=2)
        self.c1br7 = Conv1x1BnRelu(512, 1024)

        self.dwc3br8 = DWConv3x3BnRelu(1024, 1024)
        self.c1br8 = Conv1x1BnRelu(1024, 1024)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = Flatten()
        self.fc = nn.Linear(1024, classes)

    def forward(self, input):
        x = self.c3br1(input)
        x = self.dwc3br1(x)
        x = self.c1br1(x)

        x = self.dwc3br2(x)
        x = self.c1br2(x)

        x = self.dwc3br3(x)
        x = self.c1br3(x)

        x = self.dwc3br4(x)
        x = self.c1br4(x)

        x = self.dwc3br5(x)
        x = self.c1br5(x)

        x = self.dwc3br6(x)
        x = self.c1br6(x)

        for idx in range(len(self.middle5x)):
            x = self.middle5x[idx](x)

        x = self.dwc3br7(x)
        x = self.c1br7(x)

        x = self.dwc3br8(x)
        x = self.c1br8(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


class MobileNetV1(nn.Module):
    def __init__(self, in_chan=3, classes=1000):
        super(MobileNetV1, self).__init__()

        self.cfg_pt1 = [(32,64,1), (64,128,2), (128,128,1), (128,256,2), (256,256,1), (256,512,2)]
        self.cfg_pt2 = [(512,512,1) for i in range(5)]
        self.cfg_pt3 = [(512,1024,2), (1024,1024,1)]
        self.config = self.cfg_pt1 + self.cfg_pt2 + self.cfg_pt3

        self.conv3x3bnrl = Conv3x3BnRelu(in_chan, 32, strd=2, pad=1)

        self.conv_layer = nn.ModuleList()
        for cfg in self.config:
            self.conv_layer.append(DWConv3x3BnRelu(cfg[0], cfg[0], strd=cfg[2]))
            self.conv_layer.append(Conv1x1BnRelu(cfg[0], cfg[1]))
        self.conv_layer = nn.Sequential(*self.conv_layer)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = Flatten()
        self.fc = nn.Linear(1024, classes)

    def forward(self, input):
        x = self.conv3x3bnrl(input)
        x = self.conv_layer(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x



if __name__ == '__main__':
    model = MobileNetV1(in_chan=3, classes=7)
    input = torch.rand(1,3,224,224)
    output = model(input)
    print(output)










