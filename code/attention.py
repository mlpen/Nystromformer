
import torch
import torch.nn as nn
import math
import json

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

type = {}

with open("/model/config.json", "r") as f:
    config = json.load(f)
model_config = config["model"]

if model_config["attn_type"] == "softmax":
    type["softmax"] = SoftmaxAttention

elif model_config["attn_type"] == "nystrom":
    from attention_nystrom import NystromAttention
    type["nystrom"] = NystromAttention

elif model_config["attn_type"] == "reformer":
    from attention_reformer import LSHAttention
    type["reformer"] = LSHAttention

elif model_config["attn_type"] == "linformer":
    from attention_linformer import LinformerAttention
    type["linformer"] = LinformerAttention

else:
    raise Exception()
