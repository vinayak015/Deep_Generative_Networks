import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

from typing import Tuple
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Below results to 1/10000^(2*i/d_model),
        # using math.log instead of torch.log because torch.log accepts only tensors
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # add batch dimension (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        sinusoidal_embedding = x + self.pe[:, :x.size(1), :]
        dropout = self.dropout(sinusoidal_embedding)
        return dropout


class SelfAttentionBlock(nn.Module):
    """
    This self attention block also acts as multi head attention block.
    Instead of projecting the input embedding to lower dim and then calculating self attention,
    and then concatenating it back for the linear layer, we can project the input to same dim
    and then split it for multi-head attention. By doing this we are avoiding multiple matrix projections.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, mask: bool = False):
        super(SelfAttentionBlock, self).__init__()
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        assert self.head_dim * n_heads == d_model, "d_model should be divisible by num of heads"

        self.to_queries = nn.Linear(d_model, d_model, bias=False)
        self.to_keys = nn.Linear(d_model, d_model, bias=False)
        self.to_values = nn.Linear(d_model, d_model, bias=False)

        self.unify_heads = nn.Linear(d_model, d_model)

    def forward(self, x, encoder_out=None, mask=None):
        batch, time_steps, d_model = x.size()

        Q = self.to_queries(x)
        Q = Q.view(batch, time_steps, self.n_heads, self.head_dim)  # head_dim * n_heads = d_model
        if encoder_out is not None:
            # cross attention
            batch, time_steps, d_model = encoder_out.size()
            K = self.to_keys(encoder_out)
            V = self.to_values(encoder_out)
        else:
            K = self.to_keys(x)
            V = self.to_values(x)

        # split the K, Q, V for each heads
        K = K.view(batch, time_steps, self.n_heads, self.head_dim)
        V = V.view(batch, time_steps, self.n_heads, self.head_dim)

        dot = torch.einsum("btnh, bsnh -> bnts", Q, K)  # dot product between (t,h)-dim and (s,h)-dim
        if mask is not None:
            dot.masked_fill(mask == 0, -1e9)
        dot = dot / math.sqrt(self.head_dim)

        attn_wt = F.softmax(dot, dim=2)

        # self_attn = torch.einsum('bnts, btnd ->bnsd', attn_wt, V)  # out = b,n_head,time,embedding
        self_attn = attn_wt @ V.transpose(1, 2)
        # (attn_wt.transpose(1, 2) @ V.transpose(1, 2)).transpose(1, 2).reshape(batch, time_steps,
        #                                                                       self.n_heads * self.head_dim)
        # (attn_wt.transpose @ V.transpose).reshape(batch, time_steps,
        #                                                                       self.n_heads * self.head_dim)
        out = self.unify_heads(self_attn.reshape(self_attn.size(0), -1, self.n_heads * self.head_dim))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttentionBlock(d_model, n_heads)

        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.layer_norm_1 = nn.LayerNorm(self.d_model)
        self.layer_norm_2 = nn.LayerNorm(self.d_model)

        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.d_model)
        )

    def forward(self, x, encoder_out=None, mask=None):
        attended = self.attention(x, encoder_out, mask)
        x = self.layer_norm_1(attended + x)

        fc = self.fc(x)
        return self.layer_norm_2(fc + x)


class TransformerEncoder(nn.Module):
    def __init__(self, seq_len: int, n_heads: int, d_model: int, vocab_size: int, num_layers: int):
        super(TransformerEncoder, self).__init__()
        self.embedding_layer = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEmbedding(d_model=d_model, max_len=seq_len)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(num_layers)
        ])

    def forward(self, x, encoder_mask=None):
        token_embed = self.embedding_layer(x)
        out = self.positional_encoding(token_embed)

        for layer in self.layers:
            out = layer(out, mask=encoder_mask)
        return out


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TransformerDecoderBlock, self).__init__()

        self.attention = SelfAttentionBlock(d_model=d_model, n_heads=n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.drop_out = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(d_model=d_model, n_heads=n_heads)

    def forward(self, x, encoder_out=None, input_mask=None, tgt_mask=None):
        attend = self.attention(x, encoder_out=None, mask=tgt_mask)
        out = self.drop_out(self.norm(attend + x))
        out = self.transformer_block(out, encoder_out, input_mask)  # cross attention
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, seq_len, num_layer, n_heads):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model=d_model, max_len=seq_len)

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads) for _ in range(num_layer)
        ])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, encoder_out, input_mask, tgt_mask):
        token_emb = self.embedding(x)
        out = self.pos_emb(token_emb)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out, encoder_out, input_mask, tgt_mask)  # it contains cross attention layer

        out = self.fc_out(out)
        return out


# class Transformer(nn.Module):
#     def __init__(self, encoder, decoder):
#         self.encoder = encoder
#         self.decoder = decoder


class Transformer(nn.Module):
    def __init__(self, tgt_vocab_size, src_vocab_size, d_model, seq_len, num_layer, n_heads):
        super(Transformer, self).__init__()

        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = TransformerEncoder(seq_len=seq_len, n_heads=n_heads, d_model=d_model,
                                          vocab_size=src_vocab_size, num_layers=num_layer)

        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, seq_len, num_layer, n_heads)

    # def forward(self, src, tgt, encoder_mask=None, decoder_mask=None):
    #     encoder_out = self.encoder(src, encoder_mask)
    #     decoder_out = self.decoder(tgt, encoder_out, decoder_mask)
    #     return {"encoder_out": encoder_out, "decoder_out": decoder_out}

    def encoder_fwd(self, src, input_mask):
        encoder_out = self.encoder(src, input_mask)
        return encoder_out

    def decoder_fwd(self, encoder_out, tgt, input_mask, tgt_mask):
        decoder_out = self.decoder(tgt, encoder_out, input_mask, tgt_mask)
        return decoder_out

#
# src_vocab_size = 11
# target_vocab_size = 11
# num_layers = 6
# seq_length = 12
# d_model = 256
# n_heads = 4
#
#
# # let 0 be sos token and 1 be eos token
# # src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1],
# #                     [0, 2, 8, 7, 3, 4, 5, 6, 7, 2, 10, 1]])
# # target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1],
# #                        [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]])
# #
# # print(src.shape,target.shape)
# # model = Transformer(target_vocab_size, src_vocab_size, d_model, seq_length, num_layers, n_heads)
# # out = model(src, target)
# # print(out.shape)
