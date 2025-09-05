from torch import nn
from utils.plain_cnn_block import plain_cnn_block


class PlainCNN(nn.Module):
    def __init__(self, in_channels=3):
        super(PlainCNN, self).__init__()
        self.conv_block1 = plain_cnn_block(in_channels,3,16,1,1, norm=True)
        self.conv_block2 = plain_cnn_block(16,3,32,1,1, norm=True, pool=2)
        self.conv_block3 = plain_cnn_block(32,5,64,2,1, norm=True)
        self.conv_block4 = plain_cnn_block(64,7,128,3,1, norm=True, pool=2)
        self.pooling = nn.AvgPool2d(kernel_size=8)
        self.linear_block = nn.Sequential(
                nn.Linear(128, 10)
                )
    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.pooling(out).view(out.size(0), -1)
        out = self.linear_block(out)
        return out

    def _init_weights(self):
        pass
