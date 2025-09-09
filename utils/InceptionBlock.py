from torch import nn
import torch
from dropblock import DropBlock2D

class InceptionBlock(nn.Module):
    def __init__(self,  in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels=out_1x1, kernel_size=1, padding=0, stride=1)
                )
        self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, red_3x3, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(red_3x3),
                nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(out_3x3),
                nn.ReLU()
                )
        self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels, red_5x5, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(red_5x5),
                nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2, stride=1),
                nn.BatchNorm2d(out_5x5),
                nn.ReLU()
                )
        self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, pool_proj, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(pool_proj),
                nn.ReLU()
                )
#        self.drop_block = nn.Dropout2d()

    def forward(self, X):
        conv1 = self.branch1(X)
        conv2 = self.branch2(X)
        conv3 = self.branch3(X)
        conv4 = self.branch4(X)
        concat = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        return concat

