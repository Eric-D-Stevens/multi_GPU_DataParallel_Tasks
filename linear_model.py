import torch
import torch.nn as nn
from torchsummary import summary

class LinMod(nn.Module):
    def __init__(self):
        super(LinMod, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(12000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.Sigmoid(),
            nn.Linear(10,10)
        )

    def forward(self, x):
        y = self.layers(x)
        return y


if __name__ == "__main__":
    model = LinMod().cuda()
    summary(model,(1,12000))