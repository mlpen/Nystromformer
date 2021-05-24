
import torch
import torch.nn as nn
import math

class NystromAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.num_landmarks = config["num_landmarks"]
        self.seq_len = config["max_seq_len"]

        self.use_conv = "conv_kernel_size" in config
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = self.num_head, out_channels = self.num_head,
                kernel_size = (config["conv_kernel_size"], 1), padding = (config["conv_kernel_size"] // 2, 0),
                bias = False,
                groups = self.num_head)

    def forward(self, Q, K, V, mask):

        Q = Q * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        K = K * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))

        if self.num_landmarks == self.seq_len:
            attn = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            X = torch.matmul(attn, V)
        else:
            Q_landmarks = Q.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)
            K_landmarks = K.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)

            kernel_1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

        if self.use_conv:
            X += self.conv(V * mask[:, None, :, None])

        return X

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        V = 1 / (torch.max(torch.sum(torch.abs(K), dim = -2)) * torch.max(torch.sum(torch.abs(K), dim = -1))) * K.transpose(-1, -2)
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, seq_len={self.seq_len}'
