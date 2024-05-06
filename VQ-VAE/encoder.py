from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # out : 7, 7, 64

        self.pre_quant_conv = nn.Conv2d(64, 4, kernel_size=3, stride=2, padding=1)  # out: 4,4,4

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pre_quant_conv(x)

        return x
