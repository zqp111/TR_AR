# -*- encoding: utf-8 -*-
'''
@File    :   pos_GCN_model.py
@Time    :   2021/04/07 11:15:43
@Author  :   zqp 
@Version :   1.0
@Contact :   zhangqipeng@buaa.edu.cn
'''

import sklearn 
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home2/zqp/TR-AR')
from pos_aware.utils.Encoder import Encoder
from pos_aware.utils.graph_attention import MultiHeadedGraphAttention, MultiHeadedGraphAttention_R
from processor.base_method import import_class
from pos_aware.utils.GCN_others import MuiltKernelGTCN_
from pos_aware.utils.vit_local import Transformer, Attention
from einops import rearrange, repeat


class S_attention(nn.Module):
    def __init__(self, point_num, input_channel, mid_channels, heads, dropout):
        super(S_attention, self).__init__()

        self.point_num = point_num
        self.input_channel = input_channel
        self.mid_channels = mid_channels

        # 输入数据BN
        self.data_bn = nn.BatchNorm1d(self.point_num * self.input_channel)


        self.s_embedding = nn.Parameter(torch.randn(1, self.point_num, 3))

        self.input_att = MultiHeadedGraphAttention_R(3, heads, mid_channels[0], dropout = dropout)
        
        self.line_weight_Q = nn.Parameter(torch.zeros(25, 32))
        self.line_weight_K = nn.Parameter(torch.zeros(25, 32))
        self.line_weight_V = nn.Parameter(torch.zeros(25, 32))

        self.encod_bn = nn.BatchNorm1d(self.point_num)

        self.line_input_att = MultiHeadedGraphAttention(25, heads, mid_channels[0], dropout = dropout)
        self.input_layer = nn.ModuleList(
            [
                MultiHeadedGraphAttention(mid_channels[i], heads, mid_channels[i+1], dropout=dropout)
                for i in range(len(self.mid_channels)-1)
            ]
        )

    def forward(self, x):
        N, C, T, V, M = x.size()  # for NTU, (N, 3, T, 25, M)

        x = x.permute(0, 4, 3, 1, 2).contiguous()  # for NTU, (N, M, 25, 3, T)
        x = x.view(N * M, V * C, T)  # for NTU, (N*M, 75, T)
        x = self.data_bn(x)
        x = x.view(N*M, V, C, T)  #for NTU,  (N*M,25,3,T)
        x = x.permute(0, 2, 3, 1).contiguous()  # for NTU, (NM, 3, T, 25)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N*M*T, V, C)

        # x = x + self.s_embedding


        encoded_x = self.input_att(x, x, x)
        # encoded_line_x = self.line_input_att(torch.matmul(x_line, self.line_weight_Q),
        #                                 torch.matmul(x_line, self.line_weight_K),
        #                                 torch.matmul(x_line, self.line_weight_V)

        # print(encoded_x.shape)

        encoded_x = self.encod_bn(encoded_x)

        for layer in self.input_layer:
            encoded_x = layer(encoded_x, encoded_x, encoded_x) # for NTU, (NMT, 25, 256)

        return encoded_x

class Model(nn.Module):
    def __init__(self,
                graph,
                kernel_num,
                edge_weight,
                lamda,

                input_channel,
                mid_channels: list,
                layer_num,
                heads, 
                encode_size, 
                d_ff, 
                dropout,  
                position_encoding_dropout,
                point_num = 25, n_classes = 60, graph_arg={}):
        super(Model, self).__init__()

        self.point_num = point_num
        self.input_channel = input_channel
        self.mid_channels = mid_channels

        self.data_bn = nn.BatchNorm1d(self.point_num * self.input_channel)

        Graph = import_class(graph)
        self.graph = Graph(**graph_arg)
        self.A = self.graph.A

        kernel_size = self.A.shape[0]
        t_kernel = 9

        self.backBone1 = MuiltKernelGTCN_(input_channel, 32, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda)
        self.backBone2 = MuiltKernelGTCN_(32, 64, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda, stride=2)
        self.backBone3 = MuiltKernelGTCN_(64, 64, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda, stride=2)

        self.s_embedding = nn.Parameter(torch.randn(1, self.point_num, 64))
        self.input_att = MultiHeadedGraphAttention_R(64, heads, mid_channels[0], dropout = dropout)
        # self.input_att = MultiHeadedGraphAttention_R(3, heads, mid_channels[0], dropout = dropout)
        self.input_layer = nn.ModuleList(
            [
                MultiHeadedGraphAttention(mid_channels[i], heads, mid_channels[i+1], dropout=dropout)
                for i in range(len(self.mid_channels)-1)
            ]
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 75 + 1, encode_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, encode_size))

        self.cls_token2 = nn.Parameter(torch.randn(1, 1, encode_size))
        self.pos_embedding2  = nn.Parameter(torch.randn(1, self.point_num+1, encode_size))
        self.transformer = Transformer(dim = encode_size, 
                                        depth=layer_num, 
                                        heads=heads, 
                                        dim_head = encode_size, 
                                        mlp_dim=d_ff, 
                                        dropout=dropout)

        self.dropout = nn.Dropout(position_encoding_dropout)


        self.s_transformer = Transformer(dim = encode_size,
                                        depth= layer_num,
                                        heads = heads,
                                        dim_head = encode_size, 
                                        mlp_dim=d_ff, 
                                        dropout=dropout)


        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(encode_size),
            nn.Linear(encode_size, n_classes)
        )


    def forward(self, x):
        N, C, T, V, M = x.size()  # for NTU, (N, 3, T, 25, M)

        x = x.permute(0, 4, 3, 1, 2).contiguous()  # for NTU, (N, M, 25, 3, T)
        x = x.view(N * M, V * C, T)  # for NTU, (N*M, 75, T)
        x = self.data_bn(x)
        x = x.view(N*M, V, C, T)  #for NTU,  (N*M,25,3,T)
        x = x.permute(0, 2, 3, 1).contiguous()  # for NTU, (NM, 3, T, 25)
        # print(x.shape)

        x1 = self.backBone1(x)
        # print(x1.shape)
        x2 = self.backBone2(x1)
        # print(x2.shape)
        x3 = self.backBone3(x2)  # for NTU, (NM, 64, T, 25)
        # print("after backbone", x3.shape)

        _, C, T, V = x3.shape
        x = x3.permute(0, 2, 3, 1).contiguous()
        x = x.view(N*M*T, V, C) # for NTU, (NMT, 25, 64)
        # print("before encode layer", x.shape)

        # x += self.s_embedding

        encoded_x = self.input_att(x, x, x)
        for layer in self.input_layer:
            encoded_x = layer(encoded_x, encoded_x, encoded_x) # for NTU, (NMT, 25, 256)
        # print("after encode layer",encoded_x.shape)

        encoded_x = encoded_x.view(N, M, T, V, -1) # for NTU, (N, M, T, 25, 256)
        encoded_x = encoded_x.permute(0, 1, 3, 2, 4).contiguous() # for NTU, (N, M, 25, T, 256)
        encoded_x = encoded_x.view(N*M*V, T, -1)
        # print("before transformer encoder ", encoded_x.shape)

        b, n, _ = encoded_x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, encoded_x), dim=1)
        # print(x.shape, self.pos_embedding.shape)
        x += self.pos_embedding[:, :(n + 1)]
        # print(x.shape)
        x = self.dropout(x)

        x = self.transformer(x, None)
        x = x[:, 0]
        x = x.view(N*M, V, -1)
        b, n , d = x.shape
        # print("after transformer:", x.shape)

        S_cls = repeat(self.cls_token2, '() n d -> b n d', b = b)
        x = torch.cat((S_cls, x), dim=1)
        x = x + self.pos_embedding2
        x = self.dropout(x)

        x = self.s_transformer(x, None)
        x = x[:, 0]
        x = self.mlp_head(x)
        x = x.view(N, M, -1).mean(dim=1).squeeze(1)
        # print(x.shape)
        return x




if __name__ == "__main__":
    m = Model(
        graph='graph.ntu_rgb_d.Graph',
        kernel_num=8,
        edge_weight=True,
        lamda=1,
        
        input_channel=3,
        mid_channels=[64, 256],
        layer_num=4,
        heads=8, 
        encode_size=256, 
        d_ff=256, 
        dropout=0, 
        position_encoding_dropout=0,
        point_num=25, 
        n_classes=60
    ).cuda(1)
    a = torch.randn((2, 3, 300, 25, 2)).cuda(1)
    y = m(a)
    print(y.shape)


