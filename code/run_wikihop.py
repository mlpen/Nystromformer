from model_wrapper import ModelForWiKiHop
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
from dataset import WiKiHopDataset
import torch
import torch.nn as nn
import time
import os
import json
import numpy as np
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--from_cp", type = str, help = "from_cp", dest = "from_cp", default = None)
parser.add_argument("--batch_size", type = int, help = "batch size", dest = "batch_size", required = True)
parser.add_argument("--lr", type = float, help = "learning rate", dest = "lr", required = True)
parser.add_argument("--epoch", type = int, help = "epoch", dest = "epoch", required = True)
parser.add_argument("--checkpoint", type = int, help = "checkpoint path", dest = "checkpoint", required = True)
args = parser.parse_args()

with open("/model/config.json", "r") as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = config["model_checkpoints"]
wikihop_dataset_folder = config["wikihop_dataset_folder"]

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_config], indent = 4))

########################### Loading Datasets ###########################

tokenizer = utils.get_tokenizer(model_config["max_seq_len"])
tokenizer.add_special_tokens({"additional_special_tokens":["<answer>", "</answer>", "<question>", "</question>"]})
model_config["vocab_size"] = len(tokenizer.get_vocab())

train_dataset = WiKiHopDataset(tokenizer = tokenizer, folder_path = wikihop_dataset_folder, file = "train.json")
dev_datasets = WiKiHopDataset(tokenizer = tokenizer, folder_path = wikihop_dataset_folder, file = "dev.json")

data_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
num_steps_per_epoch = len(data_loader)
print(f"num_steps_per_epoch: {num_steps_per_epoch}", flush = True)

########################### Loading Model ###########################

model = ModelForWiKiHop(model_config)
print(model)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

checkpoint_path = os.path.join(checkpoint_dir, f"cp-{args.checkpoint:04}.model")
checkpoint = torch.load(checkpoint_path, map_location = 'cpu')

word_embeddings = checkpoint['model_state_dict']['model.embeddings.word_embeddings.weight'].data.numpy()
mlm_bias = checkpoint['model_state_dict']['mlm.mlm_class.bias'].data.numpy()

cp_vocab_size, embedding_dim = word_embeddings.shape
assert model_config["vocab_size"] > cp_vocab_size

extra_vocab = model_config["vocab_size"] - cp_vocab_size
extra_embeddings = np.random.normal(scale = 0.02, size = (extra_vocab, embedding_dim))
new_word_embeddings = np.concatenate([word_embeddings, extra_embeddings], axis = 0)
new_bias = np.concatenate([mlm_bias, np.zeros(extra_vocab)], axis = 0)

checkpoint['model_state_dict']['model.embeddings.word_embeddings.weight'] = torch.tensor(new_word_embeddings, dtype = torch.float)
checkpoint['model_state_dict']['mlm.mlm_class.weight'] = torch.tensor(new_word_embeddings, dtype = torch.float)
checkpoint['model_state_dict']['mlm.mlm_class.bias'] = torch.tensor(new_bias, dtype = torch.float)

missing_keys, unexpected_keys = model.module.load_state_dict(checkpoint['model_state_dict'], strict = False)
print(f"missing_keys = {missing_keys}")
print(f"unexpected_keys = {unexpected_keys}")
print("Model restored", checkpoint_path, flush = True)

if args.from_cp is not None:
    checkpoint = torch.load(args.from_cp, map_location = 'cpu')
    model.module.load_state_dict(checkpoint['model_state_dict'])
    print("Model restored", args.from_cp, flush = True)

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

amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None

start_epoch = 0
for epoch in reversed(range(args.epoch)):
    checkpoint_path = os.path.join(checkpoint_dir, f"wikihop-{args.checkpoint}-{args.lr}-{args.batch_size}-{args.epoch}-{epoch:04}.cp")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint["epoch"] + 1
        print("Model restored", checkpoint_path)
        break

########################### Running Model ###########################

log_file_name = f"wikihop-{args.checkpoint}-{args.lr}-{args.batch_size}-{args.epoch}.log"
print(f"Log file: {log_file_name}", flush = True)

if start_epoch == 0:
    log_f = open(os.path.join(checkpoint_dir, log_file_name), "w")
else:
    log_f = open(os.path.join(checkpoint_dir, log_file_name), "a+")

partition_names = ["train", "dev"]

best_scores = {name:{"current_accuracy":0.0, "best_accuracy":0.0} for name in partition_names}
accumu_steps = utils.compute_accumu_step(args.batch_size, len(device_ids), gpu_config["inst_per_gpu"])
print("accumu_steps", accumu_steps)

for epoch in range(start_epoch, args.epoch):
    for partition_name in partition_names:

        training = partition_name == "train"
        labels = {}
        predictions = {}

        if training:
            data_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
            model.train()
        else:
            data_loader = DataLoader(dev_datasets, batch_size = args.batch_size, shuffle = False)
            model.eval()

        for batch_idx, batch in enumerate(data_loader):

            t0 = time.time()

            summary = {}
            if training:
                optimizer.zero_grad()

            for percent, inputs in utils.partition_inputs(batch, accumu_steps, True):

                inputs_key = inputs["idx"].cpu().data.tolist()
                del inputs["idx"]

                if training:
                    outputs = model(**inputs)
                else:
                    with torch.no_grad():
                        outputs = model(**inputs)

                answer_positions = inputs["answer_positions"].cpu().data.tolist()
                logits = outputs["logits"].cpu().data.numpy()
                for inst_idx in range(len(inputs_key)):
                    inst_key = inputs_key[inst_idx]
                    if inst_key not in labels:
                        labels[inst_key] = answer_positions[inst_idx]
                        predictions[inst_key] = 0
                    assert labels[inst_key] == answer_positions[inst_idx]
                    predictions[inst_key] += logits[inst_idx, :]

                del outputs["logits"]

                for key in outputs:
                    outputs[key] = outputs[key].mean() * percent
                if training:
                    utils.backward(outputs["loss"], amp_scaler)

                utils.add_output_to_summary(outputs, summary)

            if training:
                utils.optimizer_step(optimizer, lr_scheduler, amp_scaler)

            t1 = time.time()

            summary["batch_idx"] = batch_idx
            summary["epoch"] = epoch
            summary["time"] = round(t1 - t0, 4)
            summary["partition_name"] = partition_name
            if training:
                summary["learning_rate"] = round(optimizer.param_groups[0]["lr"], 8)

            log_f.write(json.dumps(summary, sort_keys = True) + "\n")
            if batch_idx % 10 == 0:
                print(json.dumps(summary, sort_keys = True), flush = True)
                log_f.flush()

        corrects = []
        for inst_key in labels:
            corrects.append(labels[inst_key] == np.argmax(predictions[inst_key]).item())

        best_scores[partition_name]["current_accuracy"] = np.mean(corrects)
        if best_scores[partition_name]["current_accuracy"] > best_scores[partition_name]["best_accuracy"]:
            best_scores[partition_name]["best_accuracy"] = best_scores[partition_name]["current_accuracy"]

        log_f.write(json.dumps(best_scores[partition_name], sort_keys = True) + "\n")
        log_f.flush()
        print(json.dumps(best_scores[partition_name], sort_keys = True), flush = True)

        if training:
            dump_path = os.path.join(checkpoint_dir, f"wikihop-{args.checkpoint}-{args.lr}-{args.batch_size}-{args.epoch}-{epoch:04}.cp")
            torch.save({
                "model_state_dict":model.module.state_dict()
            }, dump_path.replace(".cp", ".model"))
            torch.save({
                "model_state_dict":model.module.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "lr_scheduler_state_dict":lr_scheduler.state_dict(),
                "epoch":epoch
            }, dump_path)
            print(f"Dump {dump_path}", flush = True)
