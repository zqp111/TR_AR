# -*- encoding: utf-8 -*-
'''
@File    :   tools.py
@Time    :   2021/04/06 21:21:04
@Author  :   zqp 
@Version :   1.0
@Contact :   zhangqipeng@buaa.edu.cn
'''

import numpy as np
import torch
from einops import rearrange, repeat
import time

def compute_squared_EDM_method4(X):
    # 获得矩阵都行和列，因为是行向量，因此一共有n个向量
    n,m = X.shape
    # 计算Gram 矩阵
    G = np.dot(X,X.T)
    # 因为是行向量，n是向量个数,沿y轴复制n倍，x轴复制一倍
    H = np.tile(np.diag(G), (n,1))
    return np.sqrt(H + H.T - 2*G)


def compute_squared_EDM_method_torch(X):
    b, n, m = X.shape
    # G = torch.matmul(X, X.transpose(1, 2))
    G = torch.einsum('bnm, bmq->bnq', (X, X.transpose(1, 2)))
    
    # print('diag:', torch.diagonal(G, dim1=-2, dim2=-1))
    H = repeat(torch.diagonal(G, dim1=-2, dim2=-1), 'b h -> b n h', n=n)
    # print('H:', H)
    # print('H.T:', H.transpose(1, 2))
    # print('G:', G)
    # print(H + H.transpose(1, 2) - 2*G)
    R = torch.sqrt(H + H.transpose(1, 2) - 2*G)*20

    return R.long()

def compute_squared_EDM_method_split(X):
    b, n, m = X.shape
    # G = torch.matmul(X, X.transpose(1, 2))
    X_split = X.unsqueeze(-2)
    # print(X_split.shape)
    G = torch.einsum('bnmc, bmqc->bnqc', (X_split, X_split.transpose(1, 2)))
    # print('G.shape', G.shape)
    # print('G:', G)
    # print('diag:', torch.diagonal(G, dim1=-2, dim2=-1))
    # print('diag.shape', torch.diagonal(G, dim1=1, dim2=2).shape)
    H = repeat(torch.diagonal(G, dim1=1, dim2=2).transpose(1, 2), 'b h c -> b n h c', n=n)
    # print('H:', H)
    # print('H.shape', H.shape)
    # print('H.T:', H.transpose(1, 2))
    
    # print(H + H.transpose(1, 2) - 2*G)
    return torch.sqrt(H + H.transpose(1, 2) - 2*G)

def compute_distances_no_loops(X, Y):
    """
    Compute the distance between each test point in X and each training point
    in Y using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = Y.shape[0]
    dists = np.zeros((num_test, num_train)) 
    dists = np.sqrt(-2*np.dot(X, Y.T) + np.sum(np.square(Y), axis = 1) + np.transpose([np.sum(np.square(X), axis = 1)]))

    return dists

def compute_distances_torch_batch(X, Y):
    """
         Compute the distance between each feature in X and each feature in Y.
         X/Y: bs, N, c

         dis = sqrt(X**2 + (Y**2)T- 2XYT)
         return:
         dis: bs, NX, NY
    """
    assert X.shape[0] == Y.shape[0], 'batch size mismatch'

    G = torch.matmul(X, Y.transpose(1, 2))
    
    HY = torch.sum(torch.square(Y), dim=2)
    HY = repeat(HY, 'b c -> b n c', n = X.shape[1])
    # print(HY.shape)

    HX = torch.sum(torch.square(X), dim=2)
    HX = repeat(HX, 'b n -> b n c', c = Y.shape[1])
    # print(HX.shape)

    dis = torch.sqrt(HX+HY-2*G)
    return dis

def compute_distances_torch_batch_head(X, Y):
    """
         Compute the distance between each feature in X and each feature in Y.
         X/Y: bs, h, N, c

         dis = sqrt(X**2 + (Y**2)T- 2XYT)
         return:
         dis: bs, NX, NY
    """
    assert X.shape[0] == Y.shape[0] and X.shape[1] == Y.shape[1] , 'batch size or head size mismatch'

    G = torch.matmul(X, Y.transpose(2, 3))
    
    HY = torch.sum(torch.square(Y), dim=3)
    HY = repeat(HY, 'b h c -> b h n c', n = X.shape[2])
    # print(HY.shape)

    HX = torch.sum(torch.square(X), dim=3)
    HX = repeat(HX, 'b h n -> b h n c', c = Y.shape[2])
    # print(HX.shape)

    dis = torch.sqrt(HX+HY-2*G)
    return dis



if __name__ == "__main__":
    # a = np.random.random((6, 3))
    # print(a)
    # print(compute_squared_EDM_method4(a))
    # print(compute_squared_EDM_method_torch(torch.from_numpy(a)))
    
    # a = np.random.random((2, 3))
    # b = np.random.random((2, 3))
    # print(a, b)
    # d = compute_distances_no_loops(a, b)
    # print(d)
    begin = time.time()
    for _ in range(1000):
        a = torch.rand((2, 2, 4, 4))
        b = torch.rand((2, 2, 3, 4))
        d = compute_distances_torch_batch_head(a, b)
        s = d.sum()
        if torch.isnan(s):
            print('nan')
    print(time.time() - begin)
    # print('d: ', d)


    