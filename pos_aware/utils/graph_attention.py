## -*- encoding: utf-8 -*-
'''
@File    :   graph_attention.py
@Time    :   2021/03/05 18:49:43
@Author  :   zqp 
@Version :   1.0
@Contact :   zhangqipeng@buaa.edu.cn
'''


import math
import torch
import torch.nn as nn
from torch import Tensor
from pos_aware.utils.tools import compute_squared_EDM_method_split, compute_squared_EDM_method_torch


def relative_attn_inner(x, y, pos_embed):
    """
    x: [batch_size, heads, length, head_dim]
    y: [batch_size, heads, length, head_dim]
    pos_embed: [length, length, head_dim]
    """
    batch_size, heads, length, _ = x.size() # [bs, head, length, hid_dim]

    xy = torch.matmul(x, y)  # [bs, head, length, length]
    # print("xy: ",xy.shape)

    x_t_r = x.reshape([heads * batch_size, 1, length, -1]) # [bs*head, 1, length, hid_dim]
    # print("x_t_r: ", x_t_r.shape)
    # print("pos_embed: ", pos_embed.shape)

    # [bs*head, 1, length, hid_dim] x [length, length, hid_dim] -> [bs*head, length, legnth, length]
    x_tz = torch.matmul(x_t_r, pos_embed).sum(1)   # [bs*head, length, length]
    x_tz_r = x_tz.reshape([batch_size, heads, length, -1])  # [bs, head, length, length]


    return xy + x_tz_r



class MultiHeadedGraphAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-p
    """

    def __init__(self, input_size, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedGraphAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(input_size, num_heads * head_size)
        self.v_layer = nn.Linear(input_size, num_heads * head_size)
        self.q_layer = nn.Linear(input_size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None, shared_rl_pe: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :param shared_rl_pe: [M, M, D]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2) # [bs, head, length, hid_size]

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        # # scores = torch.matmul(q, k.transpose(2, 3))
        #
        # # TODO? attention score + q x relative_pe
        # rl_q = q.reshape(batch_size * num_heads, 1, -1, self.head_size)  # [bs*head, 1, length, hid_size]
        # rl_content_scores = torch.matmul(rl_q, shared_rl_pe.transpose(2, 1))  # [bs*head, length, length, length]
        # rl_content_scores = rl_content_scores.sum(1, keepdim=False)  # [bs*head, length, length]
        # rl_content_scores = rl_content_scores.reshape(batch_size, num_heads, -1, self.head_size)  # [bs, head, length, length]
        #
        # scores += rl_content_scores
        # TODO, Add relative position information.
        if shared_rl_pe is not None:
            scores = relative_attn_inner(q, k.transpose(-2, -1), shared_rl_pe.transpose(-2, -1))
        else:
            scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            # print("scores: ", scores.shape)
            # print("mask: ", mask.unsqueeze(1).shape)
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)  # [bs, head, length, length]

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        # TODO?
        if shared_rl_pe is not None:
            context = relative_attn_inner(attention, v, shared_rl_pe)
        else:
            context = torch.matmul(attention, v)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)

        return output


class MultiHeadedGraphAttention_R(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-p
    """

    def __init__(self, input_size, num_heads: int, size: int, dropout: float = 0.1, r_pos = True):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedGraphAttention_R, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads


        self.r_pos = r_pos
        if self.r_pos:
            self.r_embedding = nn.Embedding(300, num_heads) 

        self.k_layer = nn.Linear(input_size, num_heads * head_size)
        self.v_layer = nn.Linear(input_size, num_heads * head_size)
        self.q_layer = nn.Linear(input_size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None, shared_rl_pe: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :param shared_rl_pe: [M, M, D]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        if self.r_pos:
            x_line = compute_squared_EDM_method_torch(v)
            x_line[x_line>=300] = 299
            r_embeded = self.r_embedding(x_line)
            


        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2) # [bs, head, length, hid_size]

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        # # scores = torch.matmul(q, k.transpose(2, 3))
        #
        # # TODO? attention score + q x relative_pe
        # rl_q = q.reshape(batch_size * num_heads, 1, -1, self.head_size)  # [bs*head, 1, length, hid_size]
        # rl_content_scores = torch.matmul(rl_q, shared_rl_pe.transpose(2, 1))  # [bs*head, length, length, length]
        # rl_content_scores = rl_content_scores.sum(1, keepdim=False)  # [bs*head, length, length]
        # rl_content_scores = rl_content_scores.reshape(batch_size, num_heads, -1, self.head_size)  # [bs, head, length, length]
        #
        # scores += rl_content_scores
        # TODO, Add relative position information.
        if shared_rl_pe is not None:
            scores = relative_attn_inner(q, k.transpose(-2, -1), shared_rl_pe.transpose(-2, -1))
        else:
            scores = torch.matmul(q, k.transpose(2, 3))

        if self.r_pos:
            # print("r_embeded:", r_embeded.shape)
            r_embeded = r_embeded.permute(0, 3, 1, 2)
            # print("scores:", scores.shape)
            # print("r_embeded:", r_embeded.shape)
            scores = scores + r_embeded


        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            # print("scores: ", scores.shape)
            # print("mask: ", mask.unsqueeze(1).shape)
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)  # [bs, head, length, length]

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        # TODO?
        if shared_rl_pe is not None:
            context = relative_attn_inner(attention, v, shared_rl_pe)
        else:
            context = torch.matmul(attention, v)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)

        return output


if __name__ == "__main__":
    m =  MultiHeadedGraphAttention(3*75, 8, 256)
    x = torch.randn((16, 300, 3*75))
    y = m(x, x, x)
    print(y.shape)