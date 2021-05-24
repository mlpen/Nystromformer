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
import pickle
import numpy as np
import random
import datetime
from collections import OrderedDict
from multiprocessing import Pool

import utils

parser = argparse.ArgumentParser()

parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
parser.add_argument("--batch_size", type = int, help = "batch size", dest = "batch_size", required = True)
parser.add_argument("--lr", type = float, help = "learning rate", dest = "lr", required = True)
parser.add_argument("--task", type = str, help = "downstream task", dest = "task", required = True)
parser.add_argument("--checkpoint", type = str, help = "checkpoint path", dest = "checkpoint", required = True)

args = parser.parse_args()

curr_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(curr_path, 'models', args.model, 'config.json'), 'r') as f:
    config = json.load(f)

model_config = config["model"]
pretraining_task_config = config["pretraining_setting"]
dataset_config = config["dataset"]
checkpoint_dir = os.path.join(curr_path, 'models', args.model, 'model')

dataset_root_folder = os.path.join(curr_path, "datasets", "ALBERT-pretrain")

with open(os.path.join(curr_path, "datasets", "GLUE", 'task_config.json'), 'r') as f:
    all_downsteam_task_config = json.load(f)

model_config["output_sen_pred"] = True

device_ids = list(range(torch.cuda.device_count()))

checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
downsteam_task = args.task

downsteam_task_config = all_downsteam_task_config["task"][downsteam_task]

downsteam_task_config["batch_size"] = args.batch_size
downsteam_task_config["learning_rate"] = args.lr

downsteam_task_config["task"] = downsteam_task
downsteam_task_config["file_path"] = os.path.join(curr_path, "datasets", "GLUE", "nlp_benchmarks.pickle")
downsteam_task_config["batches_per_epoch"] = math.ceil(downsteam_task_config["num_train_inst"] / downsteam_task_config["batch_size"])

print(f"GPU list: {device_ids}")

print(json.dumps([model_config, downsteam_task_config, dataset_config], indent = 4))

downsteam_metric = downsteam_task_config["metric"]
downsteam_task = downsteam_task_config["task"]

log_file_name = f"{args.checkpoint}-downsteam-{downsteam_task}.log".replace(" ", "_")
print(f"Log file: {log_file_name}", flush = True)
log_f = open(os.path.join(checkpoint_dir, log_file_name), "w")

########################### Load Model ###########################

data = DatasetProcessor(dataset_root_folder, dataset_config)
model = ModelWrapper(model_config)

checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
model.load_state_dict(checkpoint['model_state_dict'])
print("Model restored", checkpoint_path)

model.init_sen_class()

model = model.cuda()

model = nn.DataParallel(model, device_ids = device_ids)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = downsteam_task_config["learning_rate"],
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = 0.01)

########################### Train Model ###########################

def train():
    train = downsteam_task_config["train"]
    train_downsteam_data = BertDownsteamDatasetWrapper(data, downsteam_task_config["file_path"], downsteam_task_config["task"], train)
    train_downsteam_dataloader = torch.utils.data.DataLoader(train_downsteam_data, batch_size = downsteam_task_config["batch_size"], pin_memory = True)

    train_downsteam_dataloader_iter = iter(train_downsteam_dataloader)
    batch_idx = 0

    true_labels = []
    predictions = []

    try:
        while True:

            t0 = time.time()

            inputs = next(train_downsteam_dataloader_iter)
            for key in inputs:
                inputs[key] = inputs[key].cuda()
            curr_batch_size = inputs[key].size()[0]

            optimizer.zero_grad()

            parallel_outputs, parallel_predictions = model(inputs)

            outputs = {}
            output_batch_size = parallel_outputs["batch_size_per_device"].sum()
            for key in parallel_outputs:
                if key == "batch_size_per_device":
                    outputs["batch_size"] = output_batch_size
                    outputs[key] = parallel_outputs[key].mean()
                else:
                    outputs[key] = (parallel_outputs[key] * parallel_outputs["batch_size_per_device"]).sum() / output_batch_size

            if model_config["mixed_precision"]:
                scaler.scale(outputs["sen_loss"]).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs["sen_loss"].backward()
                optimizer.step()

            true_labels.append(inputs["sentence_label"].cpu().data.numpy())
            predictions.append(parallel_predictions.cpu().data.numpy())

            outputs = {key:round(value.data.item(), 4) for key, value in outputs.items()}

            t1 = time.time()

            outputs["batch_idx"] = batch_idx
            outputs["epoch"] = epoch
            outputs["time"] = round(t1 - t0, 4)
            outputs["learning_rate"] = round(optimizer.param_groups[0]["lr"], 8)
            outputs["partition"] = train
            outputs["checkpoint"] = os.path.basename(checkpoint_path)

            log_f.write(json.dumps(outputs, sort_keys = True) + "\n")

            if batch_idx % pretraining_task_config["batches_per_report"] == 0:
                print(json.dumps(outputs, sort_keys = True))
                log_f.flush()

            batch_idx += 1

    except StopIteration as e:
        print(f"################ {train} summary ################")

        true_labels = np.concatenate(true_labels, axis = 0).astype(np.int)
        predictions = np.concatenate(predictions, axis = 0).astype(np.int)

        aggregated_value = {
            "task":downsteam_task,
            "batch_size":downsteam_task_config["batch_size"],
            "learning_rate":downsteam_task_config["learning_rate"],
            "metric":downsteam_metric,
            "accuracy":np.mean(true_labels == predictions)
        }

        if downsteam_metric in ["F1", "rF1", "MCC"]:
            TP = np.sum((true_labels == 1) * (predictions == 1))
            FP = np.sum((true_labels == 0) * (predictions == 1))
            TN = np.sum((true_labels == 0) * (predictions == 0))
            FN = np.sum((true_labels == 1) * (predictions == 0))
            if downsteam_metric == "F1":
                aggregated_value["score"] = 2 * TP / (2 * TP + FP + FN)
            elif downsteam_metric == "rF1":
                aggregated_value["score"] = 2 * TN / (2 * TN + FN + FP)
            elif downsteam_metric == "MCC":
                aggregated_value["score"] = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            else:
                raise Exception()
        else:
            aggregated_value["score"] = aggregated_value["accuracy"]

        print(json.dumps(aggregated_value, sort_keys = True, indent = 4))
        log_f.write(json.dumps(aggregated_value, sort_keys = True) + "\n")

        print(f"################ {train} iteration end ################", flush = True)

def val():
    for val in downsteam_task_config["val"]:
        val_downsteam_data = BertDownsteamDatasetWrapper(data, downsteam_task_config["file_path"], downsteam_task_config["task"], val)
        val_downsteam_dataloader = torch.utils.data.DataLoader(val_downsteam_data, batch_size = 128, pin_memory = True)

        val_downsteam_dataloader_iter = iter(val_downsteam_dataloader)

        batch_idx = 0

        true_labels = []
        predictions = []

        try:
            while True:

                t0 = time.time()

                inputs = next(val_downsteam_dataloader_iter)
                for key in inputs:
                    inputs[key] = inputs[key].cuda()
                curr_batch_size = inputs[key].size()[0]

                parallel_outputs, parallel_predictions = model(inputs)

                outputs = {}
                output_batch_size = parallel_outputs["batch_size_per_device"].sum()
                for key in parallel_outputs:
                    if key == "batch_size_per_device":
                        outputs["batch_size"] = output_batch_size
                        outputs[key] = parallel_outputs[key].mean()
                    else:
                        outputs[key] = (parallel_outputs[key] * parallel_outputs["batch_size_per_device"]).sum() / output_batch_size

                true_labels.append(inputs["sentence_label"].cpu().data.numpy())
                predictions.append(parallel_predictions.cpu().data.numpy())

                outputs = {key:round(value.data.item(), 4) for key, value in outputs.items()}

                t1 = time.time()

                outputs["batch_idx"] = batch_idx
                outputs["epoch"] = epoch
                outputs["time"] = round(t1 - t0, 4)
                outputs["partition"] = val
                outputs["checkpoint"] = os.path.basename(checkpoint_path)

                log_f.write(json.dumps(outputs) + "\n")

                if batch_idx % pretraining_task_config["batches_per_report"] == 0:
                    print(json.dumps(outputs, sort_keys = True))
                    log_f.flush()

                batch_idx += 1

        except StopIteration as e:
            print(f"################ {val} summary ################")

            true_labels = np.concatenate(true_labels, axis = 0).astype(np.int)
            predictions = np.concatenate(predictions, axis = 0).astype(np.int)

            aggregated_value = {
                "task":downsteam_task,
                "batch_size":downsteam_task_config["batch_size"],
                "learning_rate":downsteam_task_config["learning_rate"],
                "metric":downsteam_metric,
                "accuracy":np.mean(true_labels == predictions)
            }

            if downsteam_metric in ["F1", "rF1", "MCC"]:
                TP = np.sum((true_labels == 1) * (predictions == 1))
                FP = np.sum((true_labels == 0) * (predictions == 1))
                TN = np.sum((true_labels == 0) * (predictions == 0))
                FN = np.sum((true_labels == 1) * (predictions == 0))
                if downsteam_metric == "F1":
                    aggregated_value["score"] = 2 * TP / (2 * TP + FP + FN)
                elif downsteam_metric == "rF1":
                    aggregated_value["score"] = 2 * TN / (2 * TN + FN + FP)
                elif downsteam_metric == "MCC":
                    aggregated_value["score"] = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                else:
                    raise Exception()
            else:
                aggregated_value["score"] = aggregated_value["accuracy"]

            if best_val_score[val] < aggregated_value["score"]:
                best_val_score[val] = aggregated_value["score"]

            aggregated_value["best_score"] = best_val_score[val]
            aggregated_value["val"] = val

            print(json.dumps(aggregated_value, sort_keys = True, indent = 4), flush = True)
            log_f.write(json.dumps(aggregated_value, sort_keys = True) + "\n")
            log_f.flush()

            print(f"################ {val} iteration end ################")

if model_config["mixed_precision"]:
    scaler = torch.cuda.amp.GradScaler()

idx = 0
best_val_score = {val:0.0 for val in downsteam_task_config["val"]}

for epoch in range(downsteam_task_config["epoch"]):
    model.train()
    train()

    model.eval()
    with torch.no_grad():
        val()
