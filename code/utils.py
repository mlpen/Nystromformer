import os
import torch
import argparse
import pandas as pd
import json
from collections import OrderedDict
from transformers import RobertaTokenizerFast
import math

def compute_accumu_step(batch_size, num_gpus, inst_per_gpu):
    size_per_gpu = batch_size // num_gpus
    return max(int(math.ceil(size_per_gpu / inst_per_gpu)), 1)

def get_tokenizer(max_seq_len):
    tokenizer = RobertaTokenizerFast.from_pretrained('/code/roberta-base', model_max_length = max_seq_len)
    tokenizer.model_max_length = max_seq_len
    tokenizer.init_kwargs['model_max_length'] = max_seq_len
    return tokenizer

def backward(loss, amp_scaler):
    if amp_scaler is None:
        loss.backward()
    else:
        amp_scaler.scale(loss).backward()

def optimizer_step(optimizer, lr_scheduler, amp_scaler):
    if amp_scaler is None:
        optimizer.step()
    else:
        amp_scaler.step(optimizer)
        amp_scaler.update()
    lr_scheduler.step()

def add_output_to_summary(outputs, summary):
    for key in outputs:
        if key not in summary:
            summary[key] = 0
        summary[key] = round(summary[key] + outputs[key].data.item(), 6)

def partition_inputs(inputs, num_partitions, to_cuda):
    if to_cuda:
        for key in inputs:
            inputs[key] = inputs[key].cuda()

    inputs_list = [[None, {}] for _ in range(num_partitions)]
    valid_size = None
    batch_size = None

    for key in inputs:

        if batch_size is None:
            batch_size = inputs[key].size(0)
        else:
            assert batch_size == inputs[key].size(0)

        inps = torch.chunk(inputs[key], num_partitions, dim = 0)

        if valid_size is None:
            valid_size = len(inps)
        else:
            assert valid_size == len(inps)

        for idx in range(valid_size):
            inputs_list[idx][1][key] = inps[idx]
            if inputs_list[idx][0] is None:
                inputs_list[idx][0] = inps[idx].size(0) / batch_size
            else:
                assert inputs_list[idx][0] == inps[idx].size(0) / batch_size

    return inputs_list[:valid_size]

def read_data(file):
    with open(file, "r") as f:
        data = f.read().split('\n')
    round_d = {}
    for line in data:
        try:
            values = json.loads(line.replace("'",'"'))
            round_d[values["idx"]] = values
        except Exception as e:
            print(e)
    print("Done")
    return pd.DataFrame(round_d).T

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def convert_weight_keys(weights):
    mappings = {
        "roberta.embeddings.word_embeddings.weight":"model.embeddings.word_embeddings.weight",
        "roberta.embeddings.position_embeddings.weight":"model.embeddings.position_embeddings.weight",
        "roberta.embeddings.token_type_embeddings.weight":"model.embeddings.token_type_embeddings.weight",
        "roberta.embeddings.LayerNorm.weight":"model.embeddings.norm.weight",
        "roberta.embeddings.LayerNorm.bias":"model.embeddings.norm.bias",
    }
    for idx in range(12):
        mappings[f"roberta.encoder.layer.{idx}.attention.self.query.weight"] = f"model.transformer_{idx}.mha.W_q.weight"
        mappings[f"roberta.encoder.layer.{idx}.attention.self.query.bias"] = f"model.transformer_{idx}.mha.W_q.bias"
        mappings[f"roberta.encoder.layer.{idx}.attention.self.key.weight"] = f"model.transformer_{idx}.mha.W_k.weight"
        mappings[f"roberta.encoder.layer.{idx}.attention.self.key.bias"] = f"model.transformer_{idx}.mha.W_k.bias"
        mappings[f"roberta.encoder.layer.{idx}.attention.self.value.weight"] = f"model.transformer_{idx}.mha.W_v.weight"
        mappings[f"roberta.encoder.layer.{idx}.attention.self.value.bias"] = f"model.transformer_{idx}.mha.W_v.bias"
        mappings[f"roberta.encoder.layer.{idx}.attention.output.dense.weight"] = f"model.transformer_{idx}.mha.ff.weight"
        mappings[f"roberta.encoder.layer.{idx}.attention.output.dense.bias"] = f"model.transformer_{idx}.mha.ff.bias"
        mappings[f"roberta.encoder.layer.{idx}.attention.output.LayerNorm.weight"] = f"model.transformer_{idx}.norm1.weight"
        mappings[f"roberta.encoder.layer.{idx}.attention.output.LayerNorm.bias"] = f"model.transformer_{idx}.norm1.bias"
        mappings[f"roberta.encoder.layer.{idx}.intermediate.dense.weight"] = f"model.transformer_{idx}.ff1.weight"
        mappings[f"roberta.encoder.layer.{idx}.intermediate.dense.bias"] = f"model.transformer_{idx}.ff1.bias"
        mappings[f"roberta.encoder.layer.{idx}.output.dense.weight"] = f"model.transformer_{idx}.ff2.weight"
        mappings[f"roberta.encoder.layer.{idx}.output.dense.bias"] = f"model.transformer_{idx}.ff2.bias"
        mappings[f"roberta.encoder.layer.{idx}.output.LayerNorm.weight"] = f"model.transformer_{idx}.norm2.weight"
        mappings[f"roberta.encoder.layer.{idx}.output.LayerNorm.bias"] = f"model.transformer_{idx}.norm2.bias"

    mappings[f"roberta.pooler.dense.weight"] = "model.pooler.weight"
    mappings[f"roberta.pooler.dense.bias"] = "model.pooler.bias"
    mappings[f"lm_head.dense.weight"] = "mlm.dense.weight"
    mappings[f"lm_head.dense.bias"] = "mlm.dense.bias"
    mappings[f"lm_head.layer_norm.weight"] = "mlm.norm.weight"
    mappings[f"lm_head.layer_norm.bias"] = "mlm.norm.bias"
    mappings[f"lm_head.decoder.weight"] = "mlm.mlm_class.weight"
    mappings[f"lm_head.decoder.bias"] = "mlm.mlm_class.bias"

    target_weights = OrderedDict()
    for key in weights:
        if key in mappings:
            target_weights[mappings[key]] = weights[key]
        else:
            print(f"Missing key: {key}")

    return target_weights
