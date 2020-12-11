from model_wrapper import ModelForMaskedLM
from dataset import CorpusDataset
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import json
import numpy as np
import argparse
import utils
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type = int, help = "checkpoint", dest = "checkpoint", required = True)
args = parser.parse_args()

with open("/model/config.json", "r") as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = config["model_checkpoints"]
data_folder = config["data_folder"]

print(json.dumps([model_config, pretraining_config], indent = 4))

########################### Loading Dataset ###########################

tokenizer = utils.get_tokenizer(model_config["max_seq_len"])
model_config["vocab_size"] = len(tokenizer.get_vocab())

if "dataset" not in config:
    config["dataset"] = None

dataset = CorpusDataset(folder_path = data_folder, file_json = "dev.json", files_per_batch = 16, option = config["dataset"])
data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = 0.15)
data_loader = DataLoader(dataset, batch_size = 4, collate_fn = data_collator)
pretrain_dataloader_iter = enumerate(data_loader)

########################### Loading Model ###########################

model_config["return_QKV"] = True
model = ModelForMaskedLM(model_config)
print(model)
model = model.cuda()

########################### Running Model ###########################

checkpoint_path = os.path.join(checkpoint_dir, f"cp-{args.checkpoint:04}.model")
checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
model.load_state_dict(checkpoint['model_state_dict'])
print("Model restored", checkpoint_path, flush = True)
model = model.model

model.eval()
with torch.no_grad():
    _, batch = next(pretrain_dataloader_iter)
    X, QKV_list = model(batch["input_ids"].cuda())

QKV_list = [{"Q":Q.cpu().data.numpy(), "K":K.cpu().data.numpy(), "V":V.cpu().data.numpy()} for Q, K, V in QKV_list]
with open("/model/QKV_list.pickle", "wb") as f:
    pickle.dump(QKV_list, f)
