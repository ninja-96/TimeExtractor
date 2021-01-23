import torch
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 2)
        self.bn2 = nn.BatchNorm2d(64)

        self.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc = nn.Linear(64, 10)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x