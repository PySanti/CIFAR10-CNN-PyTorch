from torch import nn
from utils.plain_cnn_block import plain_cnn_block


class PlainCNN(nn.Module):
    def __init__(self, img_size, in_channels=3):
        super(PlainCNN).__init__()
        self.conv_block1 = plain_cnn_block(in_channels,3,16,1,1)
        self.conv_block2 = plain_cnn_block(16,3,32,1,1)
        self.conv_block3 = plain_cnn_block(32,3,64,1,1)
        self.pooling = nn.AvgPool2d(img_size)
        self.linear_block = nn.Sequential(
                nn.Linear(64, 100),
                nn.Linear(100, 10)
                )
    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.pooling(out)
        out = self.linear_block(out)
        return out

    def _init_weights(self):
        pass
