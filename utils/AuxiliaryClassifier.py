import torch.nn as nn

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        # Capas del clasificador auxiliar basadas en la arquitectura original de GoogLeNet
        self.auxiliary = nn.Sequential(
            nn.AvgPool2d(kernel_size=1, stride=1), 
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.linear = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(1024, num_classes)
                )

    def forward(self, x):
        x = self.auxiliary(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)
