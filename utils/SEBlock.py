from typing import final
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, out_channels) -> None:
        super(SEBlock, self).__init__()
        self.linear_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(out_channels, 500),
                nn.ReLU(),
                nn.Linear(500, out_channels),
                nn.Sigmoid()
                )
    
    def forward(self, x):
        linear_out = self.linear_layer(x) #  -> [B, out_channels]
        linear_out = linear_out.unsqueeze(-1).unsqueeze(-1)
        return x*linear_out


