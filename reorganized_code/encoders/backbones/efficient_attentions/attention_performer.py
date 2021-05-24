
import torch
import torch.nn as nn
import math
from performer_pytorch import FastAttention

class PerformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.rp_dim = config["rp_dim"]
        self.kernel_type = config["kernel_type"]
        if self.kernel_type == "relu":
            self.attn_fn = FastAttention(dim_heads = self.head_dim, nb_features = self.rp_dim, causal = False, kernel_fn = nn.ReLU())
        elif self.kernel_type == "exp":
            self.attn_fn = FastAttention(dim_heads = self.head_dim, nb_features = self.rp_dim, causal = False, kernel_fn = torch.exp)

    def forward(self, Q, K, V, mask):
        return self.attn_fn(
            Q / math.sqrt(math.sqrt(self.head_dim)),
            K / math.sqrt(math.sqrt(self.head_dim)) * mask[:, None, :, None],
            V * mask[:, None, :, None])

    def extra_repr(self):
        return f'rp_dim={self.rp_dim}, kernel_type={self.kernel_type}'
