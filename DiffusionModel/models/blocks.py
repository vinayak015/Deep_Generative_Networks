import torch
import torch.nn as nn


def get_time_embedding(time_steps, t_emb_dim):
    factor = 10000 ** (torch.arange(0, t_emb_dim, 2).float() / t_emb_dim) /(t_emb_dim//2)
    factor = factor.to(time_steps.device)
    t_emb = time_steps.unsqueeze(-1).repeat(1, t_emb_dim//2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1).to(time_steps.device)

    return t_emb


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb, down_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.down_sample = down_sample
        self.num_layers = num_layers
        self.resnet_in = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        self.t_emb = nn.Sequential(
            nn.SiLU(), nn.Linear(t_emb, out_channels)
        )
        self.resnet_mid = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1)
        self.attn_norm = nn.GroupNorm(8, out_channels)
        self.self_attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        self.down_sample = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb):
        out = x
        out = self.resnet_in(out)
        t_emb = self.t_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        out = out + t_emb

        out = self.resnet_mid(out)
        out = out + self.residual(x)

        b, c, h, w = out.shape
        attn_in = out.reshape(b, c, h*w)
        attn_in = self.attn_norm(attn_in)
        attn_in = attn_in.transpose(1, 2)
        att_out, _ = self.self_attention(attn_in, attn_in, attn_in)
        att_out = att_out.transpose(1, 2).reshape(b, c, h, w)
        out = out + att_out

        out = self.down_sample(out)

        return out


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb, num_heads):
        super().__init__()
        self.first_resnet_block_in = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        self.first_t_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb, out_channels)
        )
        self.first_resnet_block_out = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.first_residual = nn.Conv2d(in_channels, out_channels, 1)
        self.attn_norm = nn.GroupNorm(8, out_channels)
        self.self_attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        self.second_resnet_block_in = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.second_t_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb, out_channels)
        )
        self.second_resnet_block_out = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.second_residual = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x, t_emb):
        out = x
        out = self.first_resnet_block_in(out)
        out = out + self.first_t_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        out = out + self.first_resnet_block_out(out)
        out = out + self.first_residual(x)

        b, c, h, w = out.shape
        attn_in = out.reshape(b, c, h*w)
        attn_in = self.attn_norm(attn_in)
        attn_in = attn_in.transpose(1, 2)
        att_out, _ = self.self_attention(attn_in, attn_in, attn_in)
        att_out = att_out.transpose(1, 2).reshape(b, c, h, w)

        resnet_in = att_out

        out = self.second_resnet_block_in(att_out)
        out = out + self.second_t_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        out = self.second_resnet_block_out(out)
        out = out + self.second_residual(resnet_in)

        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, t_emb, up_sample):
        super().__init__()
        self.up_sample = up_sample
        self.resnet_in = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        self.t_emb = nn.Sequential(
            nn.SiLU(), nn.Linear(t_emb, out_channels)
        )
        self.resnet_mid = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1)
        self.attn_norm = nn.GroupNorm(8, out_channels)
        self.self_attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        self.up_sample_conv = nn.ConvTranspose2d(in_channels//2, in_channels//2, 4, 2, 1) if self.up_sample else nn.Identity()

    def forward(self, x, down_out, t_emb):
        out = x
        out = self.up_sample_conv(out)
        out = torch.cat([down_out, out], dim=1)
        resnet_in = out
        out = self.resnet_in(out)
        t_emb = self.t_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        out = out+ t_emb

        out = self.resnet_mid(out)
        out = out + self.residual(resnet_in)

        b, c, h, w = out.shape
        attn_in = out.reshape(b, c, h * w)
        attn_in = self.attn_norm(attn_in)
        attn_in = attn_in.transpose(1, 2)
        att_out, _ = self.self_attention(attn_in, attn_in, attn_in)
        att_out = att_out.transpose(1, 2).reshape(b, c, h, w)
        out = out + att_out

        return out


class Unet(nn.Module):
    def __init__(self, in_channels, down_channels=[32, 64, 128, 256], mid_channels=[256, 256, 128],
                 t_emb_dim=128, down_sample=[True, True, False],):
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
            [DownBlock(down_channels[i], down_channels[i+1], t_emb_dim, down_sample=down_sample[i]) for i in range(len(down_channels)-1)]
        )

        self.mids = nn.ModuleList([
            MidBlock(mid_channels[i], mid_channels[i+1], t_emb_dim, num_heads=4) for i in range(len(mid_channels)-1)
        ])

        # Channels multiplied by 2 because of concatenation happening at UpBlock
        self.ups = nn.ModuleList([
            UpBlock(down_channels[i]*2, 16 if i==0 else down_channels[i-1], num_heads=4, t_emb=t_emb_dim, up_sample=down_sample[i]) for i in range(len(down_channels)-2, -1, -1)
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

#
# unet = Unet(1)
# rand_x = torch.randn([4, 1, 28, 28])
# rand_t = torch.randint(1, 9, size=[4])
# unet(rand_x, rand_t)
