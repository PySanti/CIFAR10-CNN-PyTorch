from torch import nn
from utils.InceptionBlock import InceptionBlock
from dropblock import DropBlock2D


class InceptionNet(nn.Module):
    def __init__(self, channels=3):
        super(InceptionNet, self).__init__()
        self.conv_layers = nn.Sequential(
                InceptionBlock(in_channels=3, out_channels=16, channel_filter_size=16),
                nn.AvgPool2d(2),
                DropBlock2D(0.3, 4),
                InceptionBlock(in_channels=16*4, out_channels=32, channel_filter_size=32),
                nn.AvgPool2d(2),
                DropBlock2D(0.3, 2),
                InceptionBlock(in_channels=32*4, out_channels=64, channel_filter_size=64),
                nn.AvgPool2d(2),
                DropBlock2D(0.3, 1),
                InceptionBlock(in_channels=64*4, out_channels=128, channel_filter_size=128),
                )
        self.linear_layers = nn.Sequential(
                nn.Linear(128*4, 1000),
                nn.ReLU(),
                nn.Dropout(0.35),
                nn.Linear(1000, 10)
                )
        self.pool = nn.AvgPool2d(kernel_size=4)

    def forward(self, x):
        out = self.conv_layers(x)
        out = self.pool(out).view(out.size(0), -1)
        out = self.linear_layers(out)
        return out

    def _init_weights(self):
        pass
