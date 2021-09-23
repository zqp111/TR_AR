import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, repeat
from .attention_conv import WindowAttention, PatchMerging
from .cross_attention import CrossAtten, Residual, PreNorm, FeedForward
import math
import torch.nn.functional as F



class MultiScaleCrossAttn(nn.Module):
    def __init__(self, in_channels=3, hdim=96, heads=3,  downscaling_factors=[16, 4, 7], window_sizes=[7, 4, 4]):
        super().__init__()
        head_dim = hdim // heads
        assert hdim % heads == 0

        self.layers = nn.ModuleList()
        for i in range(len(downscaling_factors)):
            self.patch = PatchMerging(in_channels=in_channels, out_channels=hdim, downscaling_factor=downscaling_factors[i])
            self.win_attn = Residual(PreNorm(hdim, WindowAttention(hidden_dim=hdim, heads=heads, head_dim=head_dim, window_size=window_sizes[i])))
            self.mlp_block = Residual(PreNorm(hdim, FeedForward(dim=hdim, hidden_dim=hdim)))
            self.layers.append(nn.Sequential(self.patch, self.win_attn, self.mlp_block))

        self.ln_norm = nn.LayerNorm(hdim)
        # self.ln_norm = BertLayerNorm(hdim)
        self.inner_attn = CrossAtten(hdim)
        self.mlp_block_inner = Residual(PreNorm(hdim, FeedForward(dim=hdim, hidden_dim=hdim)))

    def forward(self, x):
        """
        x: [bs, c, h, w]
        """
        # patch embedding and local window self-attention
        patch_x = []
        for layer in self.layers:
            x_ = layer(x)
            patch_x.append(x_)        
        
        x1 = patch_x[0]
        # print("x1: ", x1.shape)
        bs, h_1, w_1, hdim = x1.size()
        unfold_x = []
        for p_x in patch_x[1:]:
            # print("p_x: ", p_x.shape)
            _, p_h, p_w, _ = p_x.size()    # [bs, p_h, p_w, c]

            # TODO, how to expand
            stride = math.ceil(p_h/h_1) 
            kernel = math.ceil(p_h/h_1)
            pad = stride * h_1 - p_h

            # TODO
            # stride = 1
            # kernel = 2
            # pad = kernel - 1

            p_x = p_x.permute(0, 3, 1, 2)
            p_x = F.pad(p_x, (int(pad/2), pad-int(pad/2), int(pad/2), pad-int(pad/2))) # pad the last 2 dimensions, `(padLeft, padRight, padTop, padBottom)
            # print("kernel, stride, padding: ", kernel, stride, pad)
            # print("after pad: ", p_x.shape)  # [bs, c, p_h_pad, p_w_pad]

            # [bs, c, p_h, p_w] -> [bs, c, p_h, p_w]
            # 通过padding，展开成 h_1*w_1 的倍数
            p_x = F.unfold(p_x, kernel_size=(kernel, kernel), stride=(stride, stride), padding=0) # unfold into multiples of h_1*w_1 
            # assert p_x.size(-1) == h_1*w_1
            # print("after unfold: ", p_x.shape)   # [bs, c*kernel*kernel, p_h_pad/stride * p_w_pad/stride]
            

            # 展开成 h_1*w_1 的倍数
            p_x = p_x.reshape(bs, -1, hdim, h_1*w_1).permute(0, 3, 1, 2).contiguous() 
            # print("after reshape: ", p_x.shape)
            # exit()
           
            unfold_x.append(p_x)


        x1 = x1.reshape(bs, h_1*w_1, hdim).unsqueeze(2) # [bs, h_1*w_1, 1, hdim]

        # inner attention
        x1_norm = self.ln_norm(x1)
        neighbors_norm = self.ln_norm(torch.cat(unfold_x, dim=-2))

        out = self.inner_attn(x1_norm, neighbors_norm)
        out = out + x1
        out = self.mlp_block_inner(out)
        return out



if __name__ == "__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    msca1 = MultiScaleCrossAttn(in_channels=3, hdim=32, heads=1, downscaling_factors=[7, 2, 4], window_sizes=[2, 7, 7])
    msca2 = MultiScaleCrossAttn(in_channels=32, hdim=64, heads=1, downscaling_factors=[2, 4], window_sizes=[4, 2])
    msca3 = MultiScaleCrossAttn(in_channels=64, hdim=128, heads=2, downscaling_factors=[2, 4], window_sizes=[4, 2])
    msca4 = MultiScaleCrossAttn(in_channels=128, hdim=256, heads=4, downscaling_factors=[2, 1], window_sizes=[1, 2])
    msca5 = MultiScaleCrossAttn(in_channels=256, hdim=512, heads=8, downscaling_factors=[2, 1], window_sizes=[1, 2])
    x1 = torch.randn(5, 3, 112, 112)
    out = msca1(x1)
    print("out: ", out.shape)
    # x2 = torch.randn(5, 64, 16, 16)
    # out = msca2(x2)
    # print("out: ", out.shape)
    # x3 = torch.randn(5, 256, 4, 4)
    # out = msca3(x3)
    # print("out: ", out.shape)