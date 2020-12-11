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

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type = int, help = "checkpoint", dest = "checkpoint", default = None)
parser.add_argument("--batch_size", type = int, help = "batch size", dest = "batch_size", default = None)
parser.add_argument("--num_batch", type = int, help = "number of batches", dest = "num_batch", default = None)
args = parser.parse_args()

with open("/model/config.json", "r") as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = config["model_checkpoints"]
data_folder = config["data_folder"]

if args.batch_size is not None:
    pretraining_config["batch_size"] = args.batch_size

if args.num_batch is not None:
    pretraining_config["validate_batches_per_epoch"] = args.num_batch

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_config], indent = 4))

########################### Loading Dataset ###########################

tokenizer = utils.get_tokenizer(model_config["max_seq_len"])
model_config["vocab_size"] = len(tokenizer.get_vocab())

if "dataset" not in config:
    config["dataset"] = None

dataset = CorpusDataset(folder_path = data_folder, file_json = "dev.json", files_per_batch = 128, option = config["dataset"])
data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = 0.15)
data_loader = DataLoader(dataset, batch_size = pretraining_config["batch_size"], collate_fn = data_collator)
pretrain_dataloader_iter = enumerate(data_loader)

########################### Loading Model ###########################

model = ModelForMaskedLM(model_config)
print(model)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

########################### Running Model ###########################

accumu_steps = utils.compute_accumu_step(pretraining_config["batch_size"], len(device_ids), gpu_config["inst_per_gpu"])
print("accumu_steps", accumu_steps)

if args.checkpoint is None:
    log_f_path = os.path.join(checkpoint_dir, "validation_output.log")
    log_f = open(log_f_path, "a+")
    try:
        log = utils.read_data(log_f_path)
        start_epoch = int(log["epoch"].max().item())
    except Exception as e:
        print(e)
        start_epoch = 0
    end_epoch = pretraining_config["epoch"]
else:
    log_f_path = os.path.join(checkpoint_dir, f"validation_output-checkpoint-{args.checkpoint}.log")
    log_f = open(log_f_path, "w")
    start_epoch = args.checkpoint
    end_epoch = args.checkpoint + 1
print(f"Starts from Epoch: {start_epoch}, End at Epoch: {end_epoch}", flush = True)

for epoch in range(start_epoch, end_epoch):

    checkpoint_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.model")
    checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
    model.module.load_state_dict(checkpoint['model_state_dict'])
    print("Model restored", checkpoint_path, flush = True)

    model.eval()

    epoch_summary = {}

    with torch.no_grad():
        for batch_idx in range(pretraining_config["validate_batches_per_epoch"]):

            t0 = time.time()
            _, batch = next(pretrain_dataloader_iter)
            summary = {}
            for percent, inputs in utils.partition_inputs(batch, accumu_steps, True):
                outputs = model(**inputs)
                for key in outputs:
                    outputs[key] = outputs[key].mean() * percent
                utils.add_output_to_summary(outputs, summary)
            t1 = time.time()

            for key in summary:
                if key not in epoch_summary:
                    epoch_summary[key] = []
                epoch_summary[key].append(summary[key])

            summary["idx"] = epoch * pretraining_config["validate_batches_per_epoch"] + batch_idx
            summary["batch_idx"] = batch_idx
            summary["epoch"] = epoch
            summary["time"] = round(t1 - t0, 4)

            log_f.write(json.dumps(summary, sort_keys = True) + "\n")

            if batch_idx % pretraining_config["batches_per_report"] == 0:
                print(json.dumps(summary, sort_keys = True), flush = True)
                log_f.flush()

    for key in epoch_summary:
        epoch_summary[key] = np.mean(epoch_summary[key])

    log_f.write(json.dumps(epoch_summary, sort_keys = True) + "\n")
    log_f.flush()
    print(json.dumps(epoch_summary, sort_keys = True), flush = True)
