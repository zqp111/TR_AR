import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('/home/zqp/code/TR-AR/')
# from modules.relative_local_deberta import DisentangledLocalSelfAttention
from modules.graph_attention import MultiHeadedGraphAttention
from ST_TR.att_modules.temporal_local_attention import TransformerEncoder
from einops import rearrange, repeat

class STtrLayer(nn.Module):
    def __init__(self, hidden_size,
                ff_size, 
                drop_path_rate, 
                num_layers,
                num_heads, 
                dropout, 
                window_size, 
                emb_dropout,
                max_relative_positions,

                s_local_num_layers,
                t_local_num_layers,

                stride = 2,

                pos_att_type = 'none',
                freeze: bool = False, 
                apply_query_pool: bool = False, 
                spatial_attn: bool = False, 
                pool_layers: list = None, **kwargs):
        """[summary]

        Args:
            config (): [description]

            :param hidden_size: hidden size and size of embeddings
            :param ff_size: position-wise feed-forward layer size. (Typically this is 2*hidden_size.)
            :param num_layers: number of layers
            :param num_heads: number of heads for multi-headed attention
            :param dropout: dropout probability for Transformer layers
            :param emb_dropout: Is applied to the input (word embeddings).
        """

        super(STtrLayer, self).__init__()

        self.stride = stride

        self.s_tr = TransformerEncoder(hidden_size=hidden_size,
                                        ff_size=ff_size, 
                                        drop_path_rate=drop_path_rate, 
                                        num_layers=num_layers,
                                        num_heads=num_heads, 
                                        dropout=dropout, 
                                        window_size=window_size, 
                                        emb_dropout=emb_dropout,
                                        local_num_layers=s_local_num_layers, 
                                        max_relative_positions=max_relative_positions, 
                                        pool_layers=[])

        self.t_tr = TransformerEncoder(hidden_size=hidden_size,
                                        ff_size=ff_size, 
                                        drop_path_rate=drop_path_rate, 
                                        num_layers=num_layers,
                                        num_heads=num_heads, 
                                        dropout=dropout, 
                                        window_size=window_size, 
                                        emb_dropout=emb_dropout,
                                        local_num_layers=t_local_num_layers, 
                                        max_relative_positions=max_relative_positions, 
                                        pool_layers=[])

        self.bn = nn.BatchNorm2d(hidden_size)

        if stride == 1 :
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    hidden_size,
                    hidden_size,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(hidden_size),
            )


    def forward(self, x):
        """[summary]

        Args:
            x (tensor): shape: bs*T*25*F, Example, for input x: bs*300*25*3
        """
        assert len(x.shape) == 4, 'wrong shape of x'
        bs, T, N, F = x.shape

        
        x = x.permute(0, 3, 1, 2).contiguous() # bs F T N

        # print('x:', x.shape)
        res = self.residual(x)

        res = res.permute(0, 2, 3, 1).contiguous() 
        # print("res: ", res.shape)

        x = x.permute(0, 2, 3, 1).contiguous()
        # print('x:', x.shape)

        #  first s tr
        x = x.view(bs*T, N, F)
        x = self.s_tr(x, N, None)
        x = x.view(bs, T, N, F)


        # t tr
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bs*N, T, F)
        x = self.t_tr(x, T, None)
        
        if self.stride > 1:
            index = list(range(0, T, self.stride))
            x = x[:, index, :]
            _, T, _ = x.shape

        x = x.view(bs, N, T, F)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.bn(x) # bs, F, N, T
        x = x.permute(0, 3, 2, 1).contiguous()

        # x = x.view(bs, T, N, F)
        x = x + res
        return x


class Model(nn.Module):
    def __init__(self,
                num_heads,
                hidden_size,
                ff_size,
                num_layers,
                dropout,
                emb_dropout,
                s_local_num_layers,
                t_local_num_layers,
                drop_path_rate,
                window_size,
                max_relative_positions,
                input_channel,
                block_num,
                stride_layer: list, # 长度等于block num
                end_t_num_layer=6,
                point_num = 25,
                class_num = 60, 
                pos_att_type = 'none',
                freeze: bool = False, 
                apply_query_pool: bool = False, 
                spatial_attn: bool = True, 
                pool_layers: list = None, **kwargs):
        super(Model, self).__init__()

        assert len(stride_layer) == block_num, 'the num of stride_layer is wrong, must equal block num: {}'.format(block_num)
        
        self.data_bn = nn.BatchNorm1d(point_num * input_channel)

        self.input_att = MultiHeadedGraphAttention(input_channel, num_heads, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.layers = nn.ModuleList(
            [
                STtrLayer(hidden_size=hidden_size,
                        ff_size=ff_size, 
                        drop_path_rate=drop_path_rate, 
                        num_layers=num_layers,
                        num_heads=num_heads, 
                        dropout=dropout, 
                        window_size=window_size, 
                        emb_dropout=emb_dropout,
                        max_relative_positions=max_relative_positions,

                        stride=stride_layer[num],
                    
                        s_local_num_layers=s_local_num_layers,
                        t_local_num_layers=t_local_num_layers,
                        pos_att_type=pos_att_type,
                        freeze=freeze, 
                        apply_query_pool=apply_query_pool, 
                        spatial_attn=spatial_attn, 
                        pool_layers=pool_layers
                        ) 
                        for num in range(block_num)
            ]
        )

        # 做时序上的cls_token
        self.t_cls = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.t_end_tr = TransformerEncoder(hidden_size=hidden_size,
                                        ff_size=ff_size, 
                                        drop_path_rate=drop_path_rate, 
                                        num_layers=end_t_num_layer, #TODO
                                        num_heads=num_heads, 
                                        dropout=dropout, 
                                        window_size=window_size, 
                                        emb_dropout=emb_dropout,
                                        local_num_layers=t_local_num_layers, 
                                        max_relative_positions=max_relative_positions, 
                                        pool_layers=[])

        self.fc = nn.Linear(hidden_size, class_num)




    def forward(self, x):
        N, C, T, V, M = x.size()  # for NTU, (N, 3, T, 25, M)
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # for NTU, (N, M, 25, 3, T)
        x = x.view(N * M, V * C, T)  # for NTU, (N*M, 75, T)
        x = self.data_bn(x)
        x = x.view(N*M, V, C, T)  #for NTU,  (N*M,25,3,T)
        x = x.permute(0, 3, 1, 2).contiguous()  # for NTU, (NM, T, 25, 3)
        x = x.view(N*M*T, V, C)

        x = self.input_att(x, x, x)
        _, _, C = x.shape
        x = x.view(N*M, T, V, C)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=N*M).unsqueeze(1)
        # print("cls token shape", cls_tokens.shape)
        cls_tokens = repeat(cls_tokens, 'b () n d -> b t n d', t=T)
        # print("cls token shape", cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=2)
        # print('x',x.shape)

        for layer in self.layers:
            x = layer(x)
        # print('x',x.shape)

        x = x[:, :, 0, :]

        t_cls = repeat(self.t_cls, '() n d -> b n d', b=N*M)
        x = torch.cat((t_cls, x), dim=1)
        x = self.t_end_tr(x, T+1, None)[:, 0, :]
        x = x.view(N, M, -1).mean(dim=1)
        x = self.fc(x)

        return x



if __name__ == "__main__":

    class Config():
        num_heads = 8
        hidden_size = 128
        ff_size = 256
        num_layers = 1
        dropout = 0.3
        emb_dropout = 0.3
        s_local_num_layers = 0
        t_local_num_layers = 1
        drop_path_rate = 0.1
        window_size = 25
        max_relative_positions = 300
        input_channel = 3
        block_num = 8
        point_num = 25
        class_num = 60

    opt = Config()

    x = torch.randn((2, 3, 300, 25, 2)).cuda(1)

    m = Model(num_heads = 8,
        hidden_size = 128,
        ff_size = 256,
        num_layers = 1,
        dropout = 0.3,
        emb_dropout = 0.3,
        s_local_num_layers = 0,
        t_local_num_layers = 1,
        drop_path_rate = 0.1,
        window_size = 25,
        max_relative_positions = 300,
        input_channel = 3,
        block_num = 6,
        point_num = 25,
        class_num = 60,
        stride_layer = [2, 1, 2, 1, 2, 1],
        end_t_num_layer = 6).cuda(1)


    for name, layer in m.named_children():
        print(name)
        
    # print(dict([*m.named_children()])['layers'][0].s_tr)

    o = m(x)
    print(o.shape)
