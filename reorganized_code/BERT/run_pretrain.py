from dataset import DatasetProcessor, BertPreTrainDatasetWrapper, BertDownsteamDatasetWrapper
from model import Model, ModelWrapper

import torch
import torch.nn as nn
import sys
import time
import os
import math
import json
import copy
import argparse

import utils

parser = argparse.ArgumentParser()

parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)

args = parser.parse_args()

curr_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(curr_path, 'models', args.model, 'config.json'), 'r') as f:
    config = json.load(f)

model_config = config["model"]
pretraining_task_config = config["pretraining_setting"]
dataset_config = config["dataset"]
checkpoint_dir = os.path.join(curr_path, 'models', args.model, 'model')

dataset_root_folder = os.path.join(curr_path, "datasets", "ALBERT-pretrain")

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

checkpoint_path = utils.get_last_checkpoint(checkpoint_dir)

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_task_config, dataset_config], indent = 4))

########################### Loading Model ###########################

data = DatasetProcessor(dataset_root_folder, dataset_config)
model = ModelWrapper(model_config)
print(model)

num_parameter = 0
for weight in model.parameters():
    print(weight.size())
    size = 1
    for d in weight.size():
        size *= d
    num_parameter += size
print(f"num_parameter: {num_parameter}")

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

if "from_cp" in config and checkpoint_path is None:
    checkpoint = torch.load(os.path.join(curr_path, 'models', args.model, config["from_cp"]), map_location = 'cpu')

    cp_pos_encoding = checkpoint['model_state_dict']['model.embeddings.position_embeddings.weight']
    cp_max_seq_len = cp_pos_encoding.size(0)
    if model_config["max_seq_len"] > cp_max_seq_len:
        num_copy = model_config["max_seq_len"] // cp_max_seq_len
        checkpoint['model_state_dict']['model.embeddings.position_embeddings.weight'] = cp_pos_encoding.repeat(num_copy, 1)

    missing_keys, unexpected_keys = model.module.load_state_dict(checkpoint['model_state_dict'], strict = False)
    print(f"missing_keys = {missing_keys}")
    print(f"unexpected_keys = {unexpected_keys}")
    print("Model initialized", config["from_cp"])

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = pretraining_task_config["learning_rate"],
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = 0.01
)

pretrain_data = BertPreTrainDatasetWrapper(data)
pretrain_dataloader = torch.utils.data.DataLoader(
    pretrain_data, batch_size = pretraining_task_config["batch_size"], pin_memory = True)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = pretraining_task_config["learning_rate"],
    pct_start = pretraining_task_config["warmup"],
    anneal_strategy = "linear",
    epochs = pretraining_task_config["epoch"],
    steps_per_epoch = pretraining_task_config["batches_per_epoch"]
)

if checkpoint_path is not None:

    checkpoint = torch.load(checkpoint_path, map_location = 'cpu')

    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    start_epoch = checkpoint["epoch"] + 1
    inst_pass = checkpoint["inst_pass"]

    print("Model restored", checkpoint_path)
else:
    start_epoch = 0
    inst_pass = 0

########################### Train Model ###########################

if model_config["mixed_precision"]:
    scaler = torch.cuda.amp.GradScaler()

log_f = open(os.path.join(checkpoint_dir, "pretrain_output.log"), "a+")

model.train()

accumu_steps = pretraining_task_config["accumu_steps"]

pretrain_dataloader_iter = iter(pretrain_dataloader)
init_t = time.time()
for epoch in range(start_epoch, pretraining_task_config["epoch"]):
    for batch_idx in range(pretraining_task_config["batches_per_epoch"]):

        t0 = time.time()

        inputs = next(pretrain_dataloader_iter)
        inst_pass += list(inputs.values())[0].size()[0]

        if accumu_steps > 1:
            total_inputs = [{} for _ in range(accumu_steps)]
            for key in inputs:
                inputs[key] = inputs[key].cuda()
                inps = torch.chunk(inputs[key], accumu_steps, dim = 0)
                for idx in range(accumu_steps):
                    total_inputs[idx][key] = inps[idx]
        else:
            for key in inputs:
                inputs[key] = inputs[key].cuda()
            total_inputs = [inputs]

        optimizer.zero_grad()

        total_outputs = {}
        for idx in range(accumu_steps):

            outputs = model(total_inputs[idx])
            for key in outputs:
                outputs[key] = outputs[key].mean() / accumu_steps

            if model_config["mixed_precision"]:
                scaler.scale(outputs["total_loss"]).backward()
            else:
                outputs["total_loss"].backward()

            for key in outputs:
                if key not in total_outputs:
                    total_outputs[key] = 0
                total_outputs[key] = total_outputs[key] + outputs[key]

        if model_config["mixed_precision"]:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        lr_scheduler.step()

        for key in total_outputs:
            total_outputs[key] = round(total_outputs[key].data.item(), 4)

        t1 = time.time()

        total_outputs["idx"] = epoch * pretraining_task_config["batches_per_epoch"] + batch_idx
        total_outputs["batch_idx"] = batch_idx
        total_outputs["epoch"] = epoch
        total_outputs["time"] = round(t1 - t0, 4)
        total_outputs["inst_pass"] = inst_pass
        total_outputs["learning_rate"] = round(optimizer.param_groups[0]["lr"], 8)
        total_outputs["time_since_start"] = round(time.time() - init_t, 4)

        log_f.write(json.dumps(total_outputs, sort_keys = True) + "\n")

        if batch_idx % pretraining_task_config["batches_per_report"] == 0:
            print(json.dumps(total_outputs, sort_keys = True))
            log_f.flush()

    dump_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.cp")
    torch.save({
        "model_state_dict":model.module.state_dict(),
    }, dump_path.replace(".cp", ".model"))
    torch.save({
        "model_state_dict":model.module.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "lr_scheduler_state_dict":lr_scheduler.state_dict(),
        "epoch":epoch,
        "inst_pass":inst_pass
    }, dump_path)
    print(f"Dump {dump_path}")
