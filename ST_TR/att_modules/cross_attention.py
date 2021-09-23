import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, repeat
import math
import torch.nn.functional as F
# from .attention_conv import BertLayerNorm

class Gelu(nn.Module):
    def __init__(self):
        super(Gelu, self).__init__()

    def forward(self, x):
        """Implementation of the gelu activation function.
            For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            Gelu(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class CrossAtten(nn.Module):
    def __init__(self, hdim, dropout=0.2):
        super().__init__()

        self.scale = hdim ** -0.5
        self.to_q = nn.Linear(hdim, hdim, bias = False)
        self.to_kv = nn.Linear(hdim, 2*hdim, bias = False)

        self.attend = nn.Softmax(dim = -1)

        self.to_out = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.Dropout(dropout)
        )
        self.ln = nn.LayerNorm(hdim)

    def forward(self, q, kv):

        q, k, v = (self.to_q(q), *self.to_kv(kv).chunk(2, dim = -1))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # print("dots: ", dots.shape)
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # print("out: ", out.shape)

        return self.to_out(out)