import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        net = []

        net.append(nn.Conv2d(3, 64, 4, stride=2, padding=1))
        # net.append(nn.BatchNorm2d(64))
        net.append(nn.LeakyReLU(0.2))

        net.append(nn.Conv2d(64, 128, 4, stride=2, padding=1))
        net.append(nn.InstanceNorm2d(128))
        net.append(nn.LeakyReLU(0.2))

        net.append(nn.Conv2d(128, 256, 4, stride=2, padding=1))
        net.append(nn.InstanceNorm2d(256))
        net.append(nn.LeakyReLU(0.2))

        net.append(nn.Conv2d(256, 512, 4, stride=1, padding=1))
        net.append(nn.InstanceNorm2d(512))
        net.append(nn.LeakyReLU(0.2))

        net.append(nn.Conv2d(512, 1, 4, stride=1, padding=1))

        # net.append(nn.AdaptiveAvgPool2d((1, 1)))
        # net.append(nn.Flatten())
        # net.append(nn.Linear(512, 2))

        self.net = nn.Sequential(*net)

    def forward(self, X):
        Y = self.net(X)
        return F.avg_pool2d(Y, Y.size()[2:]).view(Y.size()[0], -1).float()


class Residual(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        net = []

        net.append(nn.Conv2d(input_channels,
                   output_channels, 3, padding=1, padding_mode="reflect"))
        net.append(nn.InstanceNorm2d(output_channels))
        net.append(nn.ReLU())

        net.append(nn.Conv2d(output_channels,
                   output_channels, 3, padding=1, padding_mode="reflect"))
        net.append(nn.InstanceNorm2d(output_channels))
        # net.append(nn.ReLU())

        self.net = nn.Sequential(*net)

    def forward(self, X):
        return self.net(X) + X


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        net = []

        net.append(nn.Conv2d(3, 64, 7, stride=1, padding=3,
                   padding_mode="reflect"))
        net.append(nn.InstanceNorm2d(64))
        net.append(nn.ReLU())

        net.append(nn.Conv2d(64, 128, 3,
                   stride=2, padding=1))
        net.append(nn.InstanceNorm2d(128))
        net.append(nn.ReLU())

        net.append(nn.Conv2d(128, 256,
                   3, stride=2, padding=1))
        net.append(nn.InstanceNorm2d(256))
        net.append(nn.ReLU())

        for i in range(9):
            net.append(Residual(256, 256))

        net.append(nn.ConvTranspose2d(
            256, 128, 3, stride=2, padding=1, output_padding=1))
        net.append(nn.InstanceNorm2d(128))
        net.append(nn.ReLU())

        net.append(nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=1, output_padding=1))
        net.append(nn.InstanceNorm2d(64))
        net.append(nn.ReLU())

        net.append(nn.Conv2d(64, 3, 7, stride=1,
                   padding=3, padding_mode="reflect"))
        net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

    def forward(self, X):
        return self.net(X)


if __name__ == "__main__":
    X = torch.rand(size=[5, 3, 200, 200])
    net = Discriminator()
    Y = net(X)
    print(Y)
    print(Y.shape)
import torch
import numpy as np
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        net = []

        net.append(nn.Conv2d(3, 64, 4, stride=2, padding=1))
        net.append(nn.BatchNorm2d(64))
        net.append(nn.LeakyReLU(0.2))

        net.append(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False))
        net.append(nn.BatchNorm2d(128))
        net.append(nn.LeakyReLU(0.2))

        net.append(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False))
        net.append(nn.BatchNorm2d(256))
        net.append(nn.LeakyReLU(0.2))

        net.append(nn.Conv2d(256, 512, 4, padding=1, bias=False))
        net.append(nn.BatchNorm2d(512))
        net.append(nn.LeakyReLU(0.2))

        # net.append(nn.Conv2d(512, 1, 4, padding=1))

        net.append(nn.AdaptiveAvgPool2d((1, 1)))
        net.append(nn.Flatten())
        net.append(nn.Linear(512, 2))

        self.net = nn.Sequential(*net)

    def forward(self, X):
        return self.net(X)
# 另一个比较怪的是Residual里面没怎么ReLu


class Residual(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        net = []

        net.append(nn.Conv2d(input_channels,
                   output_channels, 3, padding=1, padding_mode="reflect"))
        net.append(nn.BatchNorm2d(output_channels))
        net.append(nn.ReLU())

        net.append(nn.Conv2d(output_channels,
                   output_channels, 3, padding=1, padding_mode="reflect"))
        net.append(nn.BatchNorm2d(output_channels))
        # net.append(nn.ReLU())

        self.net = nn.Sequential(*net)

    def forward(self, X):
        return self.net(X) + X


class Generator(nn.Module):
    def __init__(self, base_dim=64, residual_num=7):
        super().__init__()
        net = []

        net.append(nn.Conv2d(3, base_dim, 7, padding=3,
                   padding_mode="reflect", bias=False))
        net.append(nn.BatchNorm2d(base_dim))
        net.append(nn.ReLU())

        net.append(nn.Conv2d(base_dim, base_dim*2, 3,
                   stride=2, padding=1, bias=False))
        net.append(nn.BatchNorm2d(base_dim*2))
        net.append(nn.ReLU())

        net.append(nn.Conv2d(base_dim*2, base_dim*4,
                   3, stride=2, padding=1, bias=False))
        net.append(nn.BatchNorm2d(base_dim*4))
        net.append(nn.ReLU())

        for i in range(residual_num):
            net.append(Residual(base_dim*4, base_dim*4))

        net.append(nn.ConvTranspose2d(
            base_dim*4, base_dim*2, 3, stride=2, padding=1, bias=False))
        net.append(nn.BatchNorm2d(base_dim*2))
        net.append(nn.ZeroPad2d([0, 1, 0, 1]))

        net.append(nn.ConvTranspose2d(
            base_dim*2, base_dim, 3, stride=2, padding=1, bias=False))
        net.append(nn.BatchNorm2d(base_dim))
        net.append(nn.ZeroPad2d([0, 1, 0, 1]))

        net.append(nn.Conv2d(base_dim, 3, 7,
                   padding=3, padding_mode="reflect"))

        self.net = nn.Sequential(*net)

    def forward(self, X):
        return self.net(X)


if __name__ == "__main__":
    X = torch.rand(size=[5, 3, 200, 200])
    net = Discriminator()
    Y = net(X)
    print(Y)
    print(Y.shape)
