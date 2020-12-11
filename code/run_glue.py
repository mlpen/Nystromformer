from model_wrapper import ModelForSequenceClassification
from transformers import RobertaTokenizerFast
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
from transformers import GlueDataset, GlueDataTrainingArguments
import torch
import torch.nn as nn
import time
import os
import json
import numpy as np
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, help = "batch size", dest = "batch_size", required = True)
parser.add_argument("--lr", type = float, help = "learning rate", dest = "lr", required = True)
parser.add_argument("--epoch", type = int, help = "epoch", dest = "epoch", required = True)
parser.add_argument("--task", type = str, help = "downstream task", dest = "task", required = True)
parser.add_argument("--checkpoint", type = int, help = "checkpoint path", dest = "checkpoint", required = True)
args = parser.parse_args()

with open("/model/config.json", "r") as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = config["model_checkpoints"]
glue_dataset_folder = config["glue_dataset_folder"]

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_config], indent = 4))

########################### Loading Datasets ###########################

tokenizer = utils.get_tokenizer(model_config["max_seq_len"])
model_config["vocab_size"] = len(tokenizer.get_vocab())

data_args = GlueDataTrainingArguments(
    task_name = args.task, data_dir = os.path.join(glue_dataset_folder, args.task),
    max_seq_length = model_config["max_seq_len"], overwrite_cache = True)
train_dataset = GlueDataset(data_args, tokenizer = tokenizer)
data_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = default_data_collator)
num_steps_per_epoch = len(data_loader)
print(f"num_steps_per_epoch: {num_steps_per_epoch}", flush = True)

dev_datasets = {"dev":GlueDataset(data_args, tokenizer = tokenizer, mode = "dev")}
if args.task.lower() == "mnli":
    data_args = GlueDataTrainingArguments(
        task_name = "mnli-mm", data_dir = os.path.join(glue_dataset_folder, args.task),
        max_seq_length = model_config["max_seq_len"], overwrite_cache = True)
    dev_datasets["dev-mm"] = GlueDataset(data_args, tokenizer = tokenizer, mode = "dev")
    model_config["num_classes"] = 3
else:
    model_config["num_classes"] = 2

########################### Loading Model ###########################

model = ModelForSequenceClassification(model_config)
print(model)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

checkpoint_path = os.path.join(checkpoint_dir, f"cp-{args.checkpoint:04}.model")
checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
missing_keys, unexpected_keys = model.module.load_state_dict(checkpoint['model_state_dict'], strict = False)
print(f"missing_keys = {missing_keys}")
print(f"unexpected_keys = {unexpected_keys}")
print("Model restored", checkpoint_path)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = args.lr,
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = 0.01
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = args.lr,
    pct_start = 0.05,
    anneal_strategy = "linear",
    epochs = args.epoch,
    steps_per_epoch = num_steps_per_epoch
)

if model_config["mixed_precision"]:
    amp_scaler = torch.cuda.amp.GradScaler()
else:
    amp_scaler = None

########################### Running Model ###########################

log_file_name = f"glue-{args.task}-{args.checkpoint}-{args.lr}-{args.batch_size}-{args.epoch}.log"
print(f"Log file: {log_file_name}", flush = True)
log_f = open(os.path.join(checkpoint_dir, log_file_name), "w")

partition_names = ["train"] + list(dev_datasets.keys())
best_scores = {name:{"current_accuracy":0.0, "current_F1":0.0, "best_accuracy":0.0, "best_F1":0.0} for name in partition_names}
accumu_steps = utils.compute_accumu_step(args.batch_size, len(device_ids), gpu_config["inst_per_gpu"])
print("accumu_steps", accumu_steps)

for epoch_idx in range(args.epoch):
    for partition_name in partition_names:

        training = partition_name == "train"
        labels = []
        predictions = []

        if training:
            data_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = default_data_collator)
            model.train()
        else:
            data_loader = DataLoader(dev_datasets[partition_name], batch_size = args.batch_size, shuffle = False, collate_fn = default_data_collator)
            model.eval()

        for batch_idx, batch in enumerate(data_loader):

            t0 = time.time()

            summary = {}
            if training:
                optimizer.zero_grad()

            for percent, inputs in utils.partition_inputs(batch, accumu_steps, True):

                if training:
                    outputs = model(**inputs)
                else:
                    with torch.no_grad():
                        outputs = model(**inputs)

                labels.extend(inputs["labels"].cpu().data.tolist())
                predictions.extend(outputs["sent_scores"].argmax(-1).cpu().data.tolist())
                del outputs["sent_scores"]

                for key in outputs:
                    outputs[key] = outputs[key].mean() * percent
                if training:
                    utils.backward(outputs["loss"], amp_scaler)

                utils.add_output_to_summary(outputs, summary)

            if training:
                utils.optimizer_step(optimizer, lr_scheduler, amp_scaler)

            t1 = time.time()

            summary["batch_idx"] = batch_idx
            summary["epoch"] = epoch_idx
            summary["time"] = round(t1 - t0, 4)
            summary["partition_name"] = partition_name
            if training:
                summary["learning_rate"] = round(optimizer.param_groups[0]["lr"], 8)

            log_f.write(json.dumps(summary, sort_keys = True) + "\n")
            if batch_idx % 10 == 0:
                print(json.dumps(summary, sort_keys = True), flush = True)
                log_f.flush()

        labels = np.asarray(labels)
        predictions = np.asarray(predictions)
        best_scores[partition_name]["current_accuracy"] = np.mean(labels == predictions)
        if best_scores[partition_name]["current_accuracy"] > best_scores[partition_name]["best_accuracy"]:
            best_scores[partition_name]["best_accuracy"] = best_scores[partition_name]["current_accuracy"]

        TP = np.sum((labels == 1) * (predictions == 1))
        FP = np.sum((labels == 0) * (predictions == 1))
        TN = np.sum((labels == 0) * (predictions == 0))
        FN = np.sum((labels == 1) * (predictions == 0))
        best_scores[partition_name]["current_F1"] = 2 * TP / (2 * TP + FP + FN + 1e-10)
        if best_scores[partition_name]["current_F1"] > best_scores[partition_name]["best_F1"]:
            best_scores[partition_name]["best_F1"] = best_scores[partition_name]["current_F1"]

        log_f.write(json.dumps(best_scores[partition_name], sort_keys = True) + "\n")
        log_f.flush()
        print(json.dumps(best_scores[partition_name], sort_keys = True), flush = True)
