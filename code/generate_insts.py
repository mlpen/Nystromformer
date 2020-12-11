from dataset import CorpusDataset, CorpusDatasetV2
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np
import argparse
import utils
import lz4.frame

parser = argparse.ArgumentParser()
parser.add_argument("--test_run", type = utils.str2bool, help = "checkpoint", dest = "test_run", default = None)
parser.add_argument("--from_epoch", type = int, help = "from_epoch", dest = "from_epoch", required = True)
parser.add_argument("--to_epoch", type = int, help = "to_epoch", dest = "to_epoch", required = True)
args = parser.parse_args()

with open("/model/config.json", "r") as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = config["model_checkpoints"]
data_folder = config["data_folder"]

print(json.dumps([model_config, pretraining_config], indent = 4))

########################### Loading Datasets ###########################

if "dataset" not in config:
    config["dataset"] = None

tokenizer = RobertaTokenizerFast.from_pretrained('/code/roberta-base', model_max_length = model_config["max_seq_len"])
tokenizer.model_max_length = model_config["max_seq_len"]
tokenizer.init_kwargs['model_max_length'] = model_config["max_seq_len"]
model_config["vocab_size"] = len(tokenizer.get_vocab())

if args.test_run is not None and args.test_run:
    dataset = CorpusDataset(folder_path = data_folder, file_json = "train.json", option = config["dataset"], files_per_batch = 64)
else:
    dataset = CorpusDataset(folder_path = data_folder, file_json = "train.json", option = config["dataset"])
data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = 0.15)
data_loader = DataLoader(dataset, batch_size = pretraining_config["batch_size"], shuffle = True, collate_fn = data_collator)
pretrain_dataloader_iter = enumerate(data_loader)

if not os.path.exists("/model/data"):
    os.mkdir("/model/data")

from_epoch = args.from_epoch
to_epoch = args.to_epoch

for epoch in range(from_epoch, to_epoch):
    instances = []
    for batch_idx in range(pretraining_config["batches_per_epoch"]):
        _, batch = next(pretrain_dataloader_iter)
        instances.append(batch)
        if batch_idx % pretraining_config["batches_per_report"] == 0:
            print(epoch, batch_idx, pretraining_config["batches_per_epoch"], flush = True)
    with lz4.frame.open(f"/model/data/batches-{epoch:04}.plz4", "wb") as f:
        pickle.dump(instances, f, protocol = pickle.HIGHEST_PROTOCOL)
