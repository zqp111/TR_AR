import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, repeat
from .temporal_local_attention import TransformerEncoderLayer
from .position_encoding import PositionalEncoding

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor, padding=0):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=padding)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        assert x.size(-1) == x.size(-2)
        b, c, h, w = x.shape
        # print("===== patch embedding ======== ")
        # print("before patch: ", x.shape)
        # print("downscaling_factor: ", self.downscaling_factor)
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)  # [b, new_h, new_w, c*down_fac*down_fac]
        x = self.linear(x)
        # print("after patch: ", x.shape)
        return x


class WindowAttention(nn.Module):
    def __init__(self, opts, window_size=7, hidden_size=32, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=opts.emb_dropout)
        self.enc_layer = TransformerEncoderLayer(opts,
                                                 size=hidden_size, 
                                                 ff_size=hidden_size*4, 
                                                 num_heads=num_heads, 
                                                 local_layer=True,
                                                 window_size=window_size)

    def forward(self, x):
        
        b, n_h, n_w, c = x.size()
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        assert n_h % self.window_size == 0, "n_h = {}, window_size={}".format(n_h, self.window_size)

        x = rearrange(x, 'b (nw_h w_h) (nw_w w_w) d -> b (nw_h nw_w) (w_h w_w) d',
                          w_h=self.window_size, w_w=self.window_size)
        x = x.reshape(-1, self.window_size * self.window_size, c)   # [bs*nw_h*nw_w, ws*ws, c]

        x = self.pe(x)
        x = self.emb_dropout(x)
        # print("before attention x: ", x.shape)
        x = self.enc_layer(x, None, None)
        x = rearrange(x, '(b nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) d',
                      w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)  # [bs, n_h, n_w, c]
        # print("after attention x: ", x.shape)
        return x.permute(0, 3, 1, 2)



class Backbone(nn.Module):
    def __init__(self, opts):
        super(Backbone, self).__init__()

        channels = [3, 32, 128, 512]
        downscaling_factors = [4, 2, 2, 2]
        window_sizes = [7, 7, 7, 7]
        layers = []
        for num in range(len(channels) - 1):
            layers.append(PatchMerging(channels[num], channels[num + 1], downscaling_factor=downscaling_factors[num], padding=0))
            layers.append(WindowAttention(opts, window_size=window_sizes[num], hidden_size=channels[num + 1], num_heads=4))
            # layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)



    def forward(self, x):
        x = self.layers(x)
        x = self.avgpool(x)
        return x.squeeze()


if __name__ == '__main__':
    x = torch.randn(5, 3, 112, 112)
    # m = PatchMerging(3, 32, downscaling_factor=1, padding=0)
    # out = m(x)
    # print("out: ", out.shape)

    class Config():
        hidden_size = 512
        ff_size = 2048
        num_heads = 8
        dropout = 0.1
        emb_dropout = 0.1
        num_layers = 6
        local_num_layers = 3
        use_relative = True
        max_relative_positions = 32
        window_size = 16

    opts = Config()
    backbone = Backbone(opts)
    out = backbone(x)
    print("out: ", out.shape)
