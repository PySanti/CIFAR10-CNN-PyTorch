from torch import nn

from utils.SEBlock import SEBlock


class SENet(nn.Module):
    def __init__(self) -> None:
        super(SENet, self).__init__()
        self.se_blocks = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.Conv2d(64, 108, 5, 1, 2),
            nn.BatchNorm2d(108),
            nn.ReLU(),
            SEBlock(108),
            nn.Conv2d(108, 186, 5, 1, 2),
            nn.BatchNorm2d(186),
            nn.ReLU(),
            SEBlock(186),
            nn.Conv2d(186, 230, 3, 1, 1),
            nn.BatchNorm2d(230),
            nn.ReLU(),
            SEBlock(230),
            nn.Conv2d(230, 300, 3, 1, 1),
            nn.BatchNorm2d(300),
            nn.ReLU(),
                )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear_layers = nn.Sequential(
                nn.Linear(300, 10)
                )
    def forward(self, x):
        out = self.se_blocks(x)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.linear_layers(out)
