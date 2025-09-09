from torch import nn

from utils.SEBlock import SEBlock
from dropblock import DropBlock2D


class SENet(nn.Module):
    def __init__(self) -> None:
        super(SENet, self).__init__()
        self.se_blocks = nn.Sequential(

            nn.Conv2d(3, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64, 8),
            nn.AvgPool2d(2),
            DropBlock2D(0.35, 4),

            nn.Conv2d(64, 96, 5, 1, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            SEBlock(96, 12),

            nn.Conv2d(96, 144, 5, 1, 2),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            SEBlock(144, 16),
            nn.AvgPool2d(2),
            DropBlock2D(0.6, 2),

            nn.Conv2d(144, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            SEBlock(192, 16),

            nn.Conv2d(192, 272, 3, 1, 1),
            nn.BatchNorm2d(272),
            nn.ReLU(),
            SEBlock(272, 32),
            nn.AvgPool2d(2),
            DropBlock2D(0.35, 1),
                )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear_layers = nn.Sequential(
                nn.Linear(272, 1000),
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(1000, 10)
                )
    def forward(self, x):
        out = self.se_blocks(x)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.linear_layers(out)
