import torch
import torch.nn as nn
import torch.nn.functional as F
from StableDiffusion.models.blocks import DownBlocks, UpBlocks, MidBlock
import yaml

from collections import OrderedDict


class Encoder(nn.Module):
    def __init__(self, im_channels, channels, t_emb=0, is_down_sample=[True], num_heads=0, num_sub_layers=1,
                 mid_layers=[256, 256]):
        super().__init__()
        self.conv_in = nn.Conv2d(im_channels, channels[0], 3, 1, 1, )

        self.downs = nn.ModuleList([
            DownBlocks(channels[i], channels[i + 1], t_emb, is_down_sample[i], num_heads, num_sub_layers) for i in
            range(len(channels) - 1)
        ])

        self.mids = nn.ModuleList([MidBlock(mid_layers[i], mid_layers[i + 1], t_emb, num_heads, num_sub_layers) for i in
                                   range(len(mid_layers) - 1)])

    def forward(self, x, t_emb=None):
        out = x
        out = self.conv_in(out)

        for layer in self.downs:
            out = layer(out, t_emb)

        for layer in self.mids:
            out = layer(out, t_emb)

        return out


class Decoder(nn.Module):
    def __init__(self, im_channels, channels, t_emb=0, is_up_sample=[True], num_heads=0, num_sub_layers=1,
                 is_concat=False):
        super().__init__()
        self.ups = nn.ModuleList(
            [UpBlocks(channels[i], channels[i + 1], num_heads, t_emb, num_sub_layers, is_up_sample[i], is_concat) for i
             in
             range(len(channels) - 1)])
        self.norm_out = nn.GroupNorm(8, channels[-1])
        self.decoder_out = nn.Conv2d(channels[-1], im_channels, 3, 1, 1)

    def forward(self, x):
        out = x
        for layer in self.ups:
            out = layer(out)
        out = self.norm_out(out)
        out = F.silu(out)
        out = self.decoder_out(out)

        return out


class VQVAE(nn.Module):
    def __init__(self, **args):
        super().__init__()
        args_ = args
        common_args = args_['common_args']
        im_channels = common_args.pop('im_channels')
        self.encoder_args = args_['Encoder']
        self.encoder_args['im_channels'] = im_channels
        self.decoder_args = args_['Decoder']
        self.decoder_args['im_channels'] = im_channels


        self.z_channels = common_args.pop('z_channels')
        self.code_book_size = common_args.pop('code_book_size')
        self.beta = common_args.pop('commitment_beta')
        self.codebook_weight = common_args.pop('codebook_weight')

        self.encoder = Encoder(**self.encoder_args)
        self.pre_quant_conv = nn.Sequential(
            nn.GroupNorm(8, self.encoder_args['channels'][-1]),
            nn.Conv2d(self.encoder_args['channels'][-1], self.z_channels, 3, 1, 1),
            nn.Conv2d(self.z_channels, self.z_channels, 1, )
        )
        self.embeddings = nn.Embedding(self.code_book_size, self.z_channels)

        self.post_quant_conv = nn.Sequential(
            nn.Conv2d(self.z_channels, self.z_channels, 1, ),
            nn.Conv2d(self.z_channels, self.encoder_args['mid_layers'][-1], 3, 1, 1),
        )

        self.decoder = Decoder(**self.decoder_args)

    def forward(self, input_):
        feat = self.encoder(input_)
        quant_out, quant_losses = self.quantize(feat)

        reconstruction = self.decoder(quant_out)

        return reconstruction, quant_out, quant_losses

    def quantize(self, feat):
        # self.embeddings = nn.Embedding(512, 3)
        quant_in = feat.permute(0, 2, 3, 1)  # reshaped to b, h, w, c
        b, h, w, c = quant_in.shape
        quant_in = quant_in.view(b, -1, c)

        # calculate distance between quant_in and embeddings,
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(quant_in, self.embeddings.weight.unsqueeze(0).repeat(b, 1, 1))

        # get indices of minimum distances from each rows
        min_indices = torch.argmin(dist, dim=-1)

        # select the nearest codebook vectors for sending it to decoder
        quant_out = torch.index_select(self.embeddings.weight, dim=0, index=min_indices.view(-1))

        quantize_loss = self.quantize_loss(quant_in, quant_out)

        quant_in = quant_in.reshape(b * h * w, c)

        # stop gradients to make sure gradient is propagated back to encoder
        # during fwd prop, the decoder takes quant_out as input but gradient is copied back to encoder during backprop
        quant_out = quant_in + (quant_out - quant_in).detach()

        quant_out = quant_out.reshape(b, h, w, c)
        quant_out = quant_out.permute(0, 3, 1, 2)

        return quant_out, quantize_loss

    def quantize_loss(self, quant_in, quant_out):
        # quant_in: b, h*w, c, quant_out: b*h*w, c
        quant_in = quant_in.reshape(-1, quant_in.shape[-1])

        # commitment loss: quant_in should be as close as to the quant_out
        # encoder should commit to codebook vectors, so that encoder avoids fluctuating between different codebook
        commitment_loss = ((quant_out.detach() - quant_in) ** 2).mean()

        # codebook loss: codebook should be as close as to the input
        # if codebooks vectors are considered as centroid of embeddings from encoder
        # then centroid should be as close to quant_in
        codebook_loss = ((quant_in.detach() - quant_out) ** 2).mean()

        quantize_losses = {"commitment_loss":self.codebook_weight * commitment_loss,
                           "code_book_loss":self.beta * codebook_loss}

        return quantize_losses

if __name__ == "main":
    with open('../configs/vqvae.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config

    x = torch.randn(16, 3, 28, 28)
    model = VQVAE(**config)
    out = model.loss(x)
