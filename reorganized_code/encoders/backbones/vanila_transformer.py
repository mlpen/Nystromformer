
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import namedtuple
import os
import sys
import pickle
import time

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config["dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.dropout = nn.Dropout(config["dropout_prob"])
        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask):

        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)

        with torch.cuda.amp.autocast(enabled = False):
            dot = dot.float() - 1e6 * (1 - mask.float()[:, None, None, :])
            attn = self.dropout(F.softmax(dot, dim = -1))
            attn_out = torch.matmul(attn, V.float())

        attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)
        return out

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config["dim"]
        self.hidden_dim = config["hidden_dim"]

        self.mha = Attention(config)

        self.dropout1 = nn.Dropout(p = config["dropout_prob"])
        self.norm1 = nn.LayerNorm(self.dim)

        self.ff = torch.nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim),
        )

        self.dropout2 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, X, mask):
        mha_out = self.norm1(X + self.dropout1(self.mha(X, mask)))
        mha_out = self.norm2(mha_out + self.dropout2(self.ff(mha_out)))
        return mha_out

class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.shared_weight = config["shared_weight"]
        self.num_layers = config["num_layers"]
        if self.shared_weight:
            self.encoder = Block(config)
        else:
            self.encoders = nn.ModuleList([Block(config) for _ in range(self.num_layers)])

    def forward(self, X, mask):
        if self.shared_weight:
            for _ in range(self.num_layers):
                X = self.encoder(X, mask)
        else:
            for encoder in self.encoders:
                X = encoder(X, mask)
        return X
