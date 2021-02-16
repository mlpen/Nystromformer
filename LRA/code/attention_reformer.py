
import torch
import torch.nn as nn
from transformers.modeling_reformer import LSHSelfAttention, ReformerConfig

class LSHAttention(LSHSelfAttention):
    def __init__(self, config, query, key, value):
        self.num_hash = config["num_hash"]
        reformer_config = ReformerConfig()
        reformer_config.attention_head_size = config["head_dim"]
        reformer_config.num_attention_heads = config["num_head"]
        reformer_config.attn_layers = ["lsh"]
        reformer_config.num_hashes = config["num_hash"]
        reformer_config.is_decoder = False
        reformer_config.max_position_embeddings = config["max_seq_len"]
        reformer_config.hidden_size = config["transformer_dim"]
        super().__init__(reformer_config)
        self.query_key.weight = query.weight
        self.value.weight = value.weight

    def forward(self, X, mask):
        return super().forward(hidden_states = X, attention_mask = mask).hidden_states

    def extra_repr(self):
        return f'num_hash={self.num_hash}'
