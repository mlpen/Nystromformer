
import torch
import torch.nn as nn
import math
import json
from torch.utils.checkpoint import checkpoint
import sys
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_path)

class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["dropout_prob"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V, mask):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X

class NoneAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, Q, K, V, mask):
        return V

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config["dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.attn_type = config["attn_type"]

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        if self.attn_type == "softmax":
            self.attn = SoftmaxAttention(config)
        elif self.attn_type == "none":
            self.attn = NoneAttention(config)
        elif self.attn_type == "linformer":
            from efficient_attentions.attention_linformer import LinformerAttention
            self.attn = LinformerAttention(config)
        elif self.attn_type == "reformer":
            from efficient_attentions.attention_reformer import LSHAttention
            self.attn = LSHAttention(config, self.W_q, self.W_k, self.W_v)
        elif self.attn_type == "nystrom":
            from efficient_attentions.attention_nystrom import NystromAttention
            self.attn = NystromAttention(config)
        elif self.attn_type == "performer":
            from efficient_attentions.attention_performer import PerformerAttention
            self.attn = PerformerAttention(config)
        elif self.attn_type == "linear":
            from efficient_attentions.attention_linear import LinearAttention
            self.attn = LinearAttention(config)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask):

        if self.attn_type in ["reformer"]:
            with torch.cuda.amp.autocast(enabled = False):
                attn_out = self.attn(X.float(), mask.float())
        else:
            Q = self.split_heads(self.W_q(X))
            K = self.split_heads(self.W_k(X))
            V = self.split_heads(self.W_v(X))

            with torch.cuda.amp.autocast(enabled = False):
                attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
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
