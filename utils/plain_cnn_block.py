from torch import nn
from dropblock import DropBlock2D


def plain_cnn_block(in_channels,kernel_size, out_channels, padding,stride=1, pool=False, norm=False, db=False):
    layers = []
    layers.append(nn.Conv2d(in_channels,out_channels, kernel_size,stride, padding))
    if norm:
        layers.append(nn.BatchNorm2d(out_channels))

    layers.append(nn.ReLU(inplace=True))
    if db:
        layers.append(DropBlock2D(0.3, db))
    if pool:
        layers.append(nn.AvgPool2d(pool))

    return nn.Sequential(*layers)
