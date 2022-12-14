# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
构建网络 用于生成patch 
"""

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        stride=1,
        dilation=1,
        groups=1,
    ):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvTanh2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        stride=1,
        dilation=1,
        groups=1,
    ):
        super(ConvTanh2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.tanh(self.bn(self.conv(x))) / 2 + 0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        stride=1,
        dilation=1,
        groups=1,
    ):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.size())
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(
        self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1
    ):
        # 这里有问题 每次卷积导致了形状变化
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.convx = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
    ):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class RGBD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RGBD, self).__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3 * in_channels, out_channels)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)

    def forward(self, x):
        x1 = self.dense(x)
        x1 = self.convdown(x1)
        x2 = self.sobelconv(x)
        x2 = self.convup(x2)
        return F.leaky_relu((x1 + x2), negative_slope=0.1)

class PatchNet(nn.Module):
    def __init__(self):
        super(PatchNet,self).__init__()
        ch=[4,8,16]
        self.en0=ConvLeakyRelu2d(3,ch[0])
        self.en1=ConvLeakyRelu2d(ch[0],ch[1])
        self.en2=ConvLeakyRelu2d(ch[1],ch[2])

        self.en3=ConvLeakyRelu2d(ch[2],ch[2])
        self.en4=ConvLeakyRelu2d(ch[2],ch[2])
        self.en5=ConvLeakyRelu2d(ch[2],ch[2])
        self.en6=ConvLeakyRelu2d(ch[2],ch[2])

        self.de6=ConvLeakyRelu2d(ch[2],ch[2])
        self.de5=ConvLeakyRelu2d(ch[2],ch[2])
        self.de4=ConvLeakyRelu2d(ch[2],ch[2])
        self.de3=ConvLeakyRelu2d(ch[2],ch[2])

        self.de2=ConvLeakyRelu2d(ch[2],ch[1])
        self.de1=ConvLeakyRelu2d(ch[1],ch[0])
        self.de0=ConvTanh2d(ch[0],3)
    def forward(self,random_patch):
        x=self.en0(random_patch)
        x=self.en1(x)
        x=self.en2(x)
        x=self.en3(x)
        x=self.en4(x)
        x=self.en5(x)
        x=self.en6(x)

        x=self.de6(x)
        x=self.de5(x)
        x=self.de4(x)
        x=self.de3(x)
        x=self.de2(x)
        x=self.de1(x)
        x=self.de0(x)
        return x

def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(1, 3, 300, 300).astype(np.float32))
    model = PatchNet()
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (1, 3, 300, 300), 'output shape (2,1,480,640) is expected!'
    print('test ok!')
    model.cuda()
    model.train()
    # optimizer=torch.optim.Adam(model.parameters(),lr=0.1)
    # print(model.parameters)
if __name__ == '__main__':
    unit_test()
