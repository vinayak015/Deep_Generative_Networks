import torch
import torch.nn as nn


def get_time_embedding(time_steps, t_emb_dim):
    factor = 10000 ** (torch.arange(0, t_emb_dim, 2).float() / t_emb_dim) / (t_emb_dim // 2)
    factor = factor.to(time_steps.device)
    t_emb = time_steps.unsqueeze(-1).repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1).to(time_steps.device)

    return t_emb


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb, down_sample=True, num_heads=4, num_layers=1,):
        super().__init__()
        self.down_sample = down_sample
        self.num_layers = num_layers
        self.t_emb = t_emb
        self.num_heads = num_heads
        self.resnet_in = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        if self.t_emb:
            self.t_emb = nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb, out_channels)
            )
        self.resnet_mid = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1)

        if num_heads:
            self.attn_norm = nn.GroupNorm(8, out_channels)
            self.self_attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

    def forward(self, x, t_emb):
        out = x
        out = self.resnet_in(out)
        if t_emb is not None:
            t_emb = self.t_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
            out = out + t_emb

        out = self.resnet_mid(out)
        out = out + self.residual(x)

        if self.num_heads:
            b, c, h, w = out.shape
            attn_in = out.reshape(b, c, h * w)
            attn_in = self.attn_norm(attn_in)
            attn_in = attn_in.transpose(1, 2)
            att_out, _ = self.self_attention(attn_in, attn_in, attn_in)
            att_out = att_out.transpose(1, 2).reshape(b, c, h, w)
            out = out + att_out

        return out


class DownBlocks(nn.Module):
    def __init__(self, c_in, c_out, t_emb, down_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.downblocks = nn.ModuleList([
            DownBlock(c_in if i == 0 else c_out, c_out, t_emb, down_sample, num_heads, num_layers) for i in
            range(num_layers)])
        self.is_down_sample = down_sample
        self.down_sample = nn.Conv2d(c_out, c_out, 4, 2, 1)

    def forward(self, x, t_emb):
        out = x
        for layer in self.downblocks:
            out = layer(out, t_emb)
        if self.is_down_sample:
            out = self.down_sample(out)
        return out


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb, num_heads, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = 4 if num_heads == 0 else num_heads
        self.first_resnet_block_in = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        if t_emb:
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

        self.attn_norm = nn.ModuleList([nn.GroupNorm(8, out_channels) for _ in range(num_layers)])
        self.self_attention = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, self.num_heads, batch_first=True) for _ in range(num_layers)])

        self.second_resnet_block_in = nn.ModuleList([nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        ) for _ in range(num_layers)])
        if t_emb:
            self.second_t_emb = nn.ModuleList([nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb, out_channels)
            ) for _ in range(num_layers)])
        self.second_resnet_block_out = nn.ModuleList([nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        ) for _ in range(num_layers)])
        self.second_residual = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 1) for _ in range(num_layers)])

    def forward(self, x, t_emb):
        out = x
        out = self.first_resnet_block_in(out)
        if t_emb is not None:
            out = out + self.first_t_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        out = out + self.first_resnet_block_out(out)
        out = out + self.first_residual(x)

        for layer in range(self.num_layers):
            b, c, h, w = out.shape
            attn_in = out.reshape(b, c, h * w)
            attn_in = self.attn_norm[layer](attn_in)
            attn_in = attn_in.transpose(1, 2)
            att_out, _ = self.self_attention[layer](attn_in, attn_in, attn_in)
            att_out = att_out.transpose(1, 2).reshape(b, c, h, w)

            resnet_in = att_out

            out = self.second_resnet_block_in[layer](att_out)
            if t_emb is not None:
                out = out + self.second_t_emb[layer](t_emb).unsqueeze(-1).unsqueeze(-1)
            out = self.second_resnet_block_out[layer](out)
            out = out + self.second_residual[layer](resnet_in)

        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, t_emb):
        super().__init__()
        self.num_heads = num_heads

        self.resnet_in = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        if t_emb:
            self.t_emb = nn.Sequential(
                nn.SiLU(), nn.Linear(t_emb, out_channels)
            )
        self.resnet_mid = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1)
        if num_heads:
            self.attn_norm = nn.GroupNorm(8, out_channels)
            self.self_attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

    def forward(self, x, t_emb):
        out = x

        resnet_in = out
        out = self.resnet_in(out)

        if t_emb is not None:
            t_emb = self.t_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
            out = out + t_emb

        out = self.resnet_mid(out)
        out = out + self.residual(resnet_in)

        if self.num_heads:
            b, c, h, w = out.shape
            attn_in = out.reshape(b, c, h * w)
            attn_in = self.attn_norm(attn_in)
            attn_in = attn_in.transpose(1, 2)
            att_out, _ = self.self_attention(attn_in, attn_in, attn_in)
            att_out = att_out.transpose(1, 2).reshape(b, c, h, w)
            out = out + att_out

        return out


class UpBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, t_emb, num_layers, is_up_sample, is_concat):
        super().__init__()
        self.is_up_sample = is_up_sample
        self.upblocks = nn.ModuleList(
            [UpBlock(in_channels if i == 0 else out_channels, out_channels, num_heads, t_emb) for i in
             range(num_layers)])

        if is_concat:
            self.up_sample = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1)
        else:
            self.up_sample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)



    def forward(self, x, x_down=None, t_emb=None):
        out = x
        if self.is_up_sample:
            out = self.up_sample(out)
        if x_down is not None:
            out = torch.cat([x_down, out], dim=1)
        for layer in self.upblocks:
            out = layer(out, t_emb)
        return out

