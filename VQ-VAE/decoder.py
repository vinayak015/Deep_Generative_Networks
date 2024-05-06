from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(4, 16, 3, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x, apply_sigmoid=False):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        if apply_sigmoid:
            out = F.sigmoid(out)
        return out
