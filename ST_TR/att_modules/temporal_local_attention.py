import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from .relative_deberta import DisentangledSelfAttention
from .relative_local_deberta_pool import DisentangledLocalSelfAttention
from .position_encoding import PositionalEncoding
import random
import logging


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(self, hidden_size,
                ff_size, 
                drop_path_rate, 
                num_layers,
                num_heads, 
                dropout, 
                window_size, 
                emb_dropout,
                local_num_layers, 
                max_relative_positions,
                pos_att_type = 'none',
                freeze: bool = False, 
                apply_query_pool: bool = False, 
                spatial_attn: bool = False, 
                pool_layers: list = None, **kwargs):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TransformerEncoder, self).__init__()

        # build all (num_layers) layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        # print("num_layers", num_layers)
        
        self.layers = nn.ModuleList(
            [
                # TransformerEncoderLayer(
                #     size=hidden_size,
                #     ff_size=ff_size,
                #     num_heads=num_heads,
                #     dropout=dropout,
                #     local_layer=num < local_num_layers,
                #     window_size=window_size, 
                #     spatial_attn=spatial_attn, # whether use relative position encoding
                #     query_pool=(num in pool_layers) and apply_query_pool,   # pooling on query
                #     drop_path = dpr[num],
                    
                # )
                TransformerEncoderLayer(
                    max_relative_positions = max_relative_positions, 
                    hidden_size = hidden_size,
                    size = hidden_size, 
                    ff_size = ff_size, 
                    num_heads = num_heads, 
                    dropout = dropout,
                    local_layer = num < local_num_layers, 
                    window_size = window_size, 
                    spatial_attn = spatial_attn, # whether use relative position encoding
                    query_pool = (num in pool_layers) and apply_query_pool,   # pooling on query, 
                    drop_path = dpr[num], 
                    pos_att_type=pos_att_type,
                    relative_attention=False, 
                    talking_head=False,
                    
                )
                for num in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self._output_size = hidden_size


    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        x = self.pe(embed_src)  # add position encoding to word embeddings
        x = self.emb_dropout(x)  # [bs, length, embed_size]

        for i, layer in enumerate(self.layers):
            # print("="*20 + "layer %i"%i + "="*20)
            x, mask = layer(x, mask, src_length)
        return self.layer_norm(x)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self, max_relative_positions, hidden_size, 
        size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1,
        local_layer: bool = False, window_size: int = 0, spatial_attn: bool = False, 
        query_pool: bool = False, drop_path: float = 0., pos_att_type='none',
        relative_attention=False, talking_head=False, max_position_embeddings=None,
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()
        self.window_size = window_size

        self.layer_norm1 = nn.LayerNorm(size, eps=1e-6)
        self.rel_embeddings = nn.Embedding(max_relative_positions * 2, size)
        if local_layer:
            print("lcoal attention, window_size: ", self.window_size)
        else:
            print("global attention")
        # self.src_src_att = DisentangledLocalSelfAttention(opts, local_layer, size, num_heads, 
        #                                                   window_size=self.window_size, 
        #                                                   spatial_attn=spatial_attn)
        self.src_src_att = DisentangledLocalSelfAttention(max_position_embeddings, dropout,
                    local_layer, hidden_size, num_heads, window_size, spatial_attn=spatial_attn, pos_att_type=pos_att_type,
                    relative_attention=relative_attention, talking_head=talking_head, max_relative_positions=max_relative_positions)


        self.layer_norm2 = nn.LayerNorm(size, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

        self.query_pool = query_pool
        if self.query_pool:
            self.conv = nn.Conv1d(size, size, kernel_size=3, stride=1, padding=1)
            self.bn = nn.BatchNorm1d(size)
            self.elu = nn.ELU()
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.reduction = nn.Sequential(self.conv, self.bn, self.elu, self.pool)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor, src_length: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """

        # print("before reduction, x: ", x.shape)   # [bs, length, embed_size]
        # print("before reduction, mask: ", mask.shape)   # [bs, length, embed_size]


        if mask is not None:
            if self.query_pool:
                # print("apply query pool!")
                mask = F.max_pool1d(mask.float(), 2, 2).byte()
            mask = self.get_attention_mask(mask)  # [bs, 1, query_len, key_len]

        if self.query_pool:
            x = x.permute(0, 2, 1).contiguous()
            x = self.reduction(x).permute(0, 2, 1).contiguous()

        # print("after reduction: ", x.shape)   # [bs, length, embed_size]
        shortcut = x   
        # x = self.dropout(self.src_src_att(
        #     hidden_states=self.layer_norm1(x),
        #     attention_mask=mask,
        #     return_att=False,
        #     query_states=None,
        #     relative_pos=None,
        #     rel_embeddings=self.rel_embeddings.weight))

        x = self.src_src_att(
            hidden_states=self.layer_norm1(x),
            attention_mask=mask,
            return_att=False,
            query_states=None,
            relative_pos=None,
            rel_embeddings=self.rel_embeddings.weight)
            
        h = x + shortcut

        o = h + self.feed_forward(h)
        return o, mask

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # print("extended_attention_mask: ", extended_attention_mask.shape)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
            # print("attention_mask: ", attention_mask.shape, attention_mask)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



if __name__ == "__main__":
    class Config():
        hidden_size = 512
        ff_size = 2048
        num_heads = 8
        dropout = 0.1
        emb_dropout = 0.1
        num_layers = 4
        local_num_layers = 3
        use_relative = True
        max_relative_positions = 32
        window_size = 16

    opts = Config()
    m = TransformerEncoder(opts)
    x = torch.randn(5, 100, 512)
    mask = torch.randint(0, 100, (5, 100)).ne(0)
    x_len = mask.sum(-1)
    out = m(x, x_len, mask)
    print("out: ", out.shape)