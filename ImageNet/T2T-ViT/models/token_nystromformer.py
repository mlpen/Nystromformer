"""
Replace the standard Transformer by the Nystromformer in T2T 
"""

import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath
from .transformer_block import Mlp

class NysAttention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_landmarks=64, kernel_size=0, init_option = "exact"):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = head_dim ** 0.5
        self.landmarks = num_landmarks
        self.kernel_size = kernel_size
        self.init_option = init_option

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.kernel_size > 0:
            self.conv = nn.Conv2d(
                in_channels = self.num_heads, out_channels = self.num_heads,
                kernel_size = (self.kernel_size, 1), padding = (self.kernel_size // 2, 0),
                bias = False,
                groups = self.num_heads)

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat

        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        if self.init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0. 
            V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence. 
            V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q /= self.scale

        keys_head_dim = k.size(-1)
        segs = N // self.landmarks
        if (N % self.landmarks == 0):
            keys_landmarks = k.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(dim = -2)
            queries_landmarks = q.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(dim = -2)
        else:
            num_k = (segs + 1) * self.landmarks - N
            keys_landmarks_f = k[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
            keys_landmarks_l = k[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
            keys_landmarks = torch.cat((keys_landmarks_f, keys_landmarks_l), dim = -2)

            queries_landmarks_f = q[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
            queries_landmarks_l = q[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
            queries_landmarks = torch.cat((queries_landmarks_f, queries_landmarks_l), dim = -2)

        kernel_1 = torch.nn.functional.softmax(torch.matmul(q, keys_landmarks.transpose(-1, -2)), dim = -1)
        kernel_2 = torch.nn.functional.softmax(torch.matmul(queries_landmarks, keys_landmarks.transpose(-1, -2)), dim = -1)
        kernel_3 = torch.nn.functional.softmax(torch.matmul(queries_landmarks, k.transpose(-1, -2)), dim = -1)
        x = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, v))

        if self.kernel_size > 0:
            x += self.conv(v)

        x = x.transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = v.squeeze(1) + x
        return x

class Token_nystromformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_landmarks = 64, kernel_size = 0, init_option = "exact"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = NysAttention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_landmarks=num_landmarks, kernel_size=kernel_size, init_option = init_option)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x