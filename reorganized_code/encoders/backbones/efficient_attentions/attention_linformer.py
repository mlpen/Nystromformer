
import torch
import torch.nn as nn
import math

class LinformerAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.num_head = config["num_head"]
        self.head_dim = config["head_dim"]
        self.linformer_k = config["linformer_k"]
        self.seq_len = config["max_seq_len"]

        if "projection_matrix" in config:
            self.E = config["projection_matrix"]
        else:
            projection_matrix = nn.Parameter(torch.Tensor(self.num_head, self.linformer_k, self.seq_len))
            torch.nn.init.normal_(projection_matrix, std = 0.02)
            self.E = projection_matrix
            config["projection_matrix"] = projection_matrix

    def forward(self, Q, K, V, mask):
        K = torch.matmul(self.E, K * mask[:, None, :, None])
        V = torch.matmul(self.E, V * mask[:, None, :, None])

        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)

        attn = nn.functional.softmax(dot, dim = -1)

        X = torch.matmul(attn, V)

        return X

    def extra_repr(self):
        return f'linformer_k={self.linformer_k}'
