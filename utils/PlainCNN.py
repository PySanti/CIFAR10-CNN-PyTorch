from torch import nn
from utils.plain_cnn_block import plain_cnn_block


class PlainCNN(nn.Module):
    def __init__(self, in_channels=3):
        super(PlainCNN, self).__init__()
        self.conv_block1 = plain_cnn_block(in_channels,7,32,3,1, norm=True, pool=2, db=8)
        self.conv_block2 = plain_cnn_block(32,5,64,2,1, norm=True, pool=2, db=4)
        self.conv_block3 = plain_cnn_block(64,5,128,2,1, norm=True, pool=2, db=2)
        self.conv_block4 = plain_cnn_block(128,3,256,1,1, norm=True, pool=2)
        self.conv_block5 = plain_cnn_block(256,3,512,1,1, norm=True, pool=2)
        self.linear_block = nn.Sequential(
                nn.Linear(512, 2000),
                nn.LeakyReLU(),
                nn.Dropout(0.4),
                nn.Linear(2000, 10)
                )
    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.conv_block5(out).view(out.size(0), -1)
        out = self.linear_block(out)
        return out

