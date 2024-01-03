import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

from typing import Tuple
import math

"""
References: Transformer code has been inspired by multiple works, especially from a blog by Peter Bloem.   
    Blog (Peter Bloem): https://peterbloem.nl/blog/transformers
    Pytorch website: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Kaggle code: https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
    Code from Umar Jamil: https://github.com/hkproj/pytorch-transformer/blob/main/model.py
                    
"""


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

    def forward(self, q, k, v, mask=None):
        q = self.to_queries(q)
        k = self.to_keys(k)
        v = self.to_values(v)

        # split the K, Q, V for each heads
        q = q.view(q.size(0), -1, self.n_heads, self.head_dim)
        k = k.view(k.size(0), -1, self.n_heads, self.head_dim)
        v = v.view(v.size(0), -1, self.n_heads, self.head_dim)

        attention_weights = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
        scaled_attention_weights = attention_weights / math.sqrt(self.head_dim)

        # We could also use einsum to calculate the attention weights as below
        # attention_weights = torch.einsum("btnh, bsnh -> bnts", q, k)  # dot product between (t,h)-dim and (s,h)-dim
        if mask is not None:
            attention_weights.masked_fill(mask == 0, -1e9)

        scaled_attention_weights = F.softmax(scaled_attention_weights, dim=2)

        # self_attn = torch.einsum('bnts, btnd ->bnsd', attn_wt, V)  # out = b,n_head,time,embedding
        self_attention = scaled_attention_weights @ v.permute(0, 2, 1, 3)

        out = self.unify_heads(self_attention.reshape(self_attention.size(0), -1, self.n_heads * self.head_dim))
        return out, self_attention


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

    def forward(self, q, k, v, mask=None):
        x = q.clone()
        attended, self_attention = self.attention(q, k, v, mask)
        x = self.layer_norm_1(attended + x)

        fc = self.fc(x)
        return self.layer_norm_2(fc + x), self_attention


class TransformerEncoder(nn.Module):
    def __init__(self, seq_len: int, n_heads: int, d_model: int, vocab_size: int, num_layers: int):
        super(TransformerEncoder, self).__init__()
        self.embedding_layer = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEmbedding(d_model=d_model, max_len=seq_len)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        token_embed = self.embedding_layer(x)
        out = self.positional_encoding(token_embed)
        self_attentions = {}
        for layer in self.layers:
            out, self_attention = layer(out, out, out, mask)
            self_attentions[f"layer_{layer}"] = self_attention
        return out, self_attentions


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TransformerDecoderBlock, self).__init__()

        self.attention = SelfAttentionBlock(d_model=d_model, n_heads=n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.drop_out = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(d_model=d_model, n_heads=n_heads)

    def forward(self, x, encoder_out=None, input_mask=None, tgt_mask=None):
        attend, self_attention = self.attention(x, x, x, tgt_mask)
        out = self.drop_out(self.norm(attend + x))
        out, cross_attention = self.transformer_block(out, encoder_out, encoder_out, input_mask)  # cross attention
        return out, self_attention, cross_attention


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
        self_attentions = {}
        cross_attentions = {}
        for layer in self.layers:
            out, self_attention, cross_attention = layer(out, encoder_out, input_mask,
                                                         tgt_mask)  # it contains cross attention layer
            self_attentions[f"layer_{layer}"] = self_attention
            cross_attentions[f"layer_{layer}"] = cross_attention

        out = self.fc_out(out)
        return out, self_attentions, cross_attentions


class Transformer(nn.Module):
    def __init__(self, tgt_vocab_size, src_vocab_size, d_model, seq_len, num_layer, n_heads):
        super(Transformer, self).__init__()

        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = TransformerEncoder(seq_len=seq_len, n_heads=n_heads, d_model=d_model,
                                          vocab_size=src_vocab_size, num_layers=num_layer)

        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, seq_len, num_layer, n_heads)

    def encoder_fwd(self, src, input_mask):
        encoder_out, self_attentions = self.encoder(src, input_mask)
        return encoder_out, self_attentions

    def decoder_fwd(self, encoder_out, tgt, input_mask, tgt_mask):
        decoder_out, self_attentions, cross_attentions = self.decoder(tgt, encoder_out, input_mask, tgt_mask)
        return decoder_out, self_attentions, cross_attentions
