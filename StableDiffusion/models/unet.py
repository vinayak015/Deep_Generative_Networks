import torch.nn as nn
from blocks import DownBlocks, MidBlock, UpBlocks, get_time_embedding


class Unet(nn.Module):
    def __init__(self, in_channels, down_channels=[32, 64, 128, 256], mid_channels=[256, 256, 128],
                 t_emb_dim=128, down_sample=[True, True, False], ):
        super().__init__()
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.t_emb_dim = t_emb_dim
        self.down_sample = down_sample

        self.t_projection = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        self.up_sample = list(reversed(self.down_channels))
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], 3, 1, 1)

        self.downs = nn.ModuleList(
            [DownBlocks(down_channels[i], down_channels[i + 1], t_emb_dim, down_sample=down_sample[i], num_layers=2) for i in
             range(len(down_channels) - 1)]
        )

        self.mids = nn.ModuleList([
            MidBlock(mid_channels[i], mid_channels[i + 1], t_emb_dim, num_heads=4, num_layers=2) for i in range(len(mid_channels) - 1)
        ])

        # Channels multiplied by 2 because of concatenation happening at UpBlock
        self.ups = nn.ModuleList([
            UpBlocks(down_channels[i] * 2, 16 if i == 0 else down_channels[i - 1], num_heads=4, t_emb=t_emb_dim,
                    is_up_sample=down_sample[i], num_layers=2) for i in range(len(down_channels) - 2, -1, -1)
        ])

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x, t):
        out = self.conv_in(x)
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_projection(t_emb)

        out_down = []
        for down in self.downs:
            out_down.append(out)
            out = down(out, t_emb)

        for mid in self.mids:
            out = mid(out, t_emb)

        for up in self.ups:
            down = out_down.pop()
            out = up(out, down, t_emb)

        out = self.norm_out(out)
        # out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out