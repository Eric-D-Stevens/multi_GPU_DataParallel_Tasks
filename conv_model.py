import torch
import torch.nn as nn
from torchsummary import summary

class ConvMod(nn.Module):
    def __init__(self):
        super(ConvMod, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1,
                    out_channels=50,
                    kernel_size=200,
                    stride=1,
                    padding=100),
    
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )


        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=50,
                    out_channels=30,
                    kernel_size=100,
                    stride=1,
                    padding=50),
    
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )


        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=30,
                    out_channels=10,
                    kernel_size=50,
                    stride=1,
                    padding=25),
    
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5)
        )



        self.linear = nn.Linear(2000,100)
        self.sig = nn.Sigmoid()
        self.out = nn.Linear(100,10)


    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = y.reshape(y.size(0),-1)
        y = self.linear(y)
        y = self.sig(y)
        y = self.out(y)
        return y


if __name__ == "__main__":
    model = ConvMod().cuda()
    summary(model,(1,12000))