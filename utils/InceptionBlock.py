from numpy import outer
from torch import nn
import torch

class InceptionBlock(nn.Module):
    def __init__(self,  in_channels, out_channels, naive=True, channel_filter_size=16):
        super(InceptionBlock, self).__init__()
        self.naive = naive;
        self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, channel_filter_size, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(channel_filter_size),
                nn.ReLU()
                )
        self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, channel_filter_size, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(channel_filter_size),
                nn.ReLU(),
                nn.Conv2d(channel_filter_size, out_channels, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )
        self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels, channel_filter_size, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(channel_filter_size),
                nn.ReLU(),
                nn.Conv2d(channel_filter_size, out_channels, kernel_size=5, padding=2, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )
        self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, channel_filter_size, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(channel_filter_size),
                nn.ReLU()
                )

    def forward(self, X):
        conv1 = self.branch1(X)
        conv2 = self.branch2(X)
        conv3 = self.branch3(X)
        conv4 = self.branch4(X)
        return torch.cat([conv1, conv2, conv3, conv4], dim=1)




