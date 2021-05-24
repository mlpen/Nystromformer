import os
import sys

curr_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

from LRA.model_wrapper import ModelForSC
import torch
import copy
import time
import pandas as pd
import json
import numpy as np

model_config = {
    "mixed_precision":True,
    "shared_weight":True,
    "embedding_dim":256,
    "dim":256,
    "hidden_dim":1024,
    "head_dim":64,
    "num_head":4,
    "num_layers":6,
    "vocab_size":512,
    "max_seq_len":64,
    "dropout_prob":0.1,
    "pooling_mode":"MEAN",
    "num_classes":10,
}
attn_config = {
    "softmax":{"attn_type":"softmax"},
    "linformer":{"attn_type":"linformer", "linformer_k":256},
    "reformer":{"attn_type":"reformer", "num_hash":2},
    "performer":{"attn_type":"performer", "rp_dim":256, "kernel_type":"relu"},
}

def func(model, batch_size, seq_len):
    input_ids = torch.randint(0, 512, (batch_size, seq_len)).long().cuda()
    labels = torch.randint(0, 10, (batch_size, )).long().cuda()
    masks = torch.ones(batch_size, seq_len).float().cuda()
    out = model(input_ids, masks, labels)
    out["loss"].mean().backward()

num_iter = 20

for attn_type in attn_config:
    print(f"attn_type={attn_type}")
    results = {}
    for log_seq_len in reversed(range(7, 13)):
        seq_len = int(2 ** log_seq_len)

        config = copy.deepcopy(model_config)
        config.update(attn_config[attn_type])
        config["max_seq_len"] = seq_len
        config["hashcode_len"] = log_seq_len
        config["hash_code_len"] = log_seq_len

        batch_size = 1
        try:
            torch.cuda.reset_peak_memory_stats()
            model = ModelForSC(config).cuda()
            while (True):
                func(model, batch_size, seq_len)
                batch_size = batch_size * 2
        except Exception as e:
            if not str(e).startswith("CUDA out of memory"):
                print(e)
        finally:
            del model
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        for _ in range(2):
            if batch_size > 1:
                batch_size = batch_size // 2
        if batch_size == 0:
            continue

        print(f"seq_len={seq_len}, batch_size={batch_size}, ", end = "")

        torch.cuda.reset_peak_memory_stats()

        model = ModelForSC(config).cuda()
        func(model, batch_size, seq_len)

        time_list = []
        for _ in range(num_iter):
            torch.cuda.synchronize()
            t0 = time.time()
            func(model, batch_size, seq_len)
            torch.cuda.synchronize()
            t1 = time.time()
            time_list.append((t1 - t0) / batch_size)

        per_inst_time_avg = np.mean(time_list) * 1000
        per_inst_time_std = np.std(time_list) * 1000
        memory_per_inst = torch.cuda.max_memory_allocated() / batch_size / 1024 / 1024

        results[seq_len] = {
            "batch_size":batch_size,
            "per_inst_time_avg (ms)":round(per_inst_time_avg, 3),
            "per_inst_time_std (ms)":round(per_inst_time_std, 3),
            "memory_per_inst (MB)":round(memory_per_inst, 3),
        }

        print(seq_len)
        print(results[seq_len])

        del model

        torch.cuda.empty_cache()

    with open(f"{attn_type}.json", "w") as f:
        json.dump(results, f, indent = 4)
