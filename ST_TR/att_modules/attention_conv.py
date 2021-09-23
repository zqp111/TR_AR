import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, repeat


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor, padding=0):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=padding)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        assert x.size(-1) == x.size(-2)
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)  # [b, new_h, new_w, c*down_fac*down_fac]
        x = self.linear(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, hidden_dim, heads, head_dim, window_size,               
                 relative_pos_embedding=None):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        # self.relative_pos_embedding = relative_pos_embedding
        self.to_qkv = nn.Linear(hidden_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, hidden_dim)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        b, n_h, n_w, _, h = *x.shape, self.heads
        # print("window attn: ", x.shape)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        assert n_h % self.window_size == 0, "n_h = {}, window_size={}".format(n_h, self.window_size)

        q, k, v = map(lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)
        # i,j 分别表示 window 里 features 的个数，也就是 (w_h*w_w)
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # print("attn: ", attn.shape)
        # print("relative_position_bias: ", relative_position_bias.shape)
        attn = attn + relative_position_bias.unsqueeze(1).unsqueeze(0)


        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)
        return out




if __name__ == "__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    win_sttn1 = WindowAttention(hidden_dim=64, heads=4, head_dim=32, window_size=7)
    win_sttn3 = WindowAttention(hidden_dim=96, heads=3, head_dim=32, window_size=4)
    win_sttn2 = WindowAttention(hidden_dim=96, heads=3, head_dim=32, window_size=4)
    x1 = torch.randn(50, 14, 14, 64)
    out1 = win_sttn1(x1)
    # x2 = torch.randn(50, 56, 56, 96).cuda()
    # out2 = win_sttn2(x2)
    # x3 = torch.randn(50, 32, 32, 96).cuda()
    # out3 = win_sttn3(x3)
    print(out1.shape)
    # print(out2.shape)
    # print(out3.shape)
