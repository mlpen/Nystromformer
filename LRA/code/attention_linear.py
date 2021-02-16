
import torch
import torch.nn as nn
import math

class LinearAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

    def forward(self, Q, K, V, mask):
        Q = (nn.functional.elu(Q) + 1) / math.sqrt(math.sqrt(Q.size(2)))
        K = (nn.functional.elu(K) + 1) * mask[:, None, :, None] / math.sqrt(math.sqrt(K.size(2)))
        V = V * mask[:, None, :, None]

        X = torch.matmul(Q, torch.matmul(torch.transpose(K, -2, -1), V))

        return X
