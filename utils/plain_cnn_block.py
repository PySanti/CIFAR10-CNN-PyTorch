from torch import nn


def plain_cnn_block(in_channels,kernel_size, out_channels, padding,stride=1, pool=False, norm=False):
    layers = []
    layers.append(nn.Conv2d(in_channels,out_channels, kernel_size,stride, padding))
    if norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    if pool:
        layers.append(nn.AvgPool2d(pool))
    return nn.Sequential(*layers)
