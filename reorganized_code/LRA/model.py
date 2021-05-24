
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.checkpoint import checkpoint

import os
import sys
curr_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

from encoders.backbones.efficient_attention import Attention

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config["embedding_dim"] == config["dim"]

        self.dim = config["embedding_dim"]

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)

        self.position_embeddings = nn.Embedding(config["max_seq_len"], config["embedding_dim"])
        torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)

        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()

        X_token = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos

        X = self.dropout(X)

        return X

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config["dim"])
        self.mha = Attention(config)
        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["dim"])

        self.mlpblock = nn.Sequential(
            nn.Linear(config["dim"], config["hidden_dim"]),
            nn.GELU(),
            torch.nn.Dropout(p = config["dropout_prob"]),
            nn.Linear(config["hidden_dim"], config["dim"]),
            torch.nn.Dropout(p = config["dropout_prob"])
        )

    def forward(self, X, mask):
        X = self.dropout1(self.mha(self.norm1(X), mask)) + X
        X = self.mlpblock(self.norm2(X)) + X
        return X

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.shared_weight = config["shared_weight"]

        self.embeddings = Embeddings(config)

        if self.shared_weight:
            self.encoder = Block(config)
        else:
            self.encoders = nn.ModuleList([Block(config) for _ in range(self.num_layers)])

        self.norm = nn.LayerNorm(config["dim"])

    def forward(self, input_ids, mask = None):

        X = self.embeddings(input_ids)

        if mask is None:
            mask = torch.ones_like(input_ids)

        if self.shared_weight:
            for _ in range(self.num_layers):
                X = self.encoder(X, mask)
        else:
            for encoder in self.encoders:
                X = encoder(X, mask)

        X = self.norm(X) * mask[:, :, None]

        return X
