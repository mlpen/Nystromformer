from dataset import DatasetProcessor, BertPreTrainDatasetWrapper, BertDownsteamDatasetWrapper
from model import Model, ModelWrapper

import argparse
import torch
import torch.nn as nn
import sys
import time
import os
import math
import json
import copy
import numpy as np

import utils

parser = argparse.ArgumentParser()

parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
parser.add_argument("--checkpoint", type = int, help = "checkpoint", dest = "checkpoint", default = None)
parser.add_argument("--batch_size", type = int, help = "batch size", dest = "batch_size", default = None)
parser.add_argument("--num_batch", type = int, help = "number of batches", dest = "num_batch", default = None)

args = parser.parse_args()

curr_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(curr_path, 'models', args.model, 'config.json'), 'r') as f:
    config = json.load(f)

model_config = config["model"]
pretraining_task_config = config["pretraining_setting"]
dataset_config = config["dataset"]
checkpoint_dir = os.path.join(curr_path, 'models', args.model, 'model')

dataset_root_folder = os.path.join(curr_path, "datasets", "ALBERT-pretrain")

if args.batch_size is not None:
    pretraining_task_config["batch_size"] = args.batch_size

if args.num_batch is not None:
    pretraining_task_config["validate_batches_per_epoch"] = args.num_batch

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_task_config, dataset_config], indent = 4))

########################### Loading Model ###########################

data = DatasetProcessor(dataset_root_folder, dataset_config, train = False)
model = ModelWrapper(model_config)
print(model)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

pretrain_data = BertPreTrainDatasetWrapper(data)
pretrain_dataloader = torch.utils.data.DataLoader(
    pretrain_data, batch_size = pretraining_task_config["batch_size"], pin_memory = True)

pretrain_dataloader_iter = iter(pretrain_dataloader)

########################### Validate Model ###########################

if args.checkpoint is None:
    log_f_path = os.path.join(checkpoint_dir, "validation_output.log")
    try:
        log = utils.read_data(log_f_path)
        start_epoch = int(log["epoch"].max().item())
    except Exception as e:
        print(e)
        start_epoch = 0
    print(f"Starts from Epoch: {start_epoch}")
    end_epoch = pretraining_task_config["epoch"]
    log_f = open(log_f_path, "a+")
else:
    start_epoch = args.checkpoint
    end_epoch = start_epoch + 1
    log_f_path = os.path.join(checkpoint_dir, f"validation_output-{args.checkpoint}.log")
    log_f = open(log_f_path, "w")

for epoch in range(start_epoch, end_epoch):

    checkpoint_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.model")
    checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
    model.module.load_state_dict(checkpoint['model_state_dict'])
    print("Model restored", checkpoint_path)

    model.eval()

    cumulative_outputs = {}

    with torch.no_grad():
        for batch_idx in range(pretraining_task_config["validate_batches_per_epoch"]):

            t0 = time.time()

            inputs = next(pretrain_dataloader_iter)
            for key in inputs:
                inputs[key] = inputs[key].cuda()

            outputs = model(inputs)
            for key in outputs:
                outputs[key] = outputs[key].mean()

            for key in outputs:
                outputs[key] = round(outputs[key].data.item(), 4)

            t1 = time.time()

            for key in outputs:
                if key not in cumulative_outputs:
                    cumulative_outputs[key] = []
                cumulative_outputs[key].append(outputs[key])

            outputs["idx"] = epoch * pretraining_task_config["validate_batches_per_epoch"] + batch_idx
            outputs["batch_idx"] = batch_idx
            outputs["epoch"] = epoch
            outputs["time"] = round(t1 - t0, 4)

            if batch_idx % pretraining_task_config["batches_per_report"] == 0:
                print(json.dumps(outputs, sort_keys = True))

    for key in cumulative_outputs:
        cumulative_outputs[key] = np.mean(cumulative_outputs[key])
    cumulative_outputs["epoch"] = epoch

    log_f.write(json.dumps(cumulative_outputs, sort_keys = True) + "\n")
    print(json.dumps(cumulative_outputs, sort_keys = True))

log_f.close()
