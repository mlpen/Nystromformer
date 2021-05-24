
import pickle
import torch
import torch.nn as nn
import numpy as np
import random
import sys
import time
import os
import bz2
import math
import json
import datetime
import matplotlib.pyplot as plt
import math
import copy
from collections import OrderedDict

import json
import tensorflow as tf
import sys
from multiprocessing import Pool
import gc

import datasets.ALBERT.tokenization as tokenization

class BertPreTrainDatasetWrapper(torch.utils.data.IterableDataset):
    def __init__(self, data_processor):
        super(BertPreTrainDatasetWrapper).__init__()
        self.data_processor = data_processor
    def __iter__(self):
        return self.data_processor.pretrain_task_generator()

class BertDownsteamDatasetWrapper(torch.utils.data.IterableDataset):
    def __init__(self, data_processor, file_path, task, partition, shuffle = True):
        super(BertDownsteamDatasetWrapper).__init__()
        self.data_processor = data_processor
        self.file_path = file_path
        self.task = task
        self.partition = partition
        self.shuffle = shuffle

    def __iter__(self):
        return self.data_processor.downsteam_task_generator(
            self.file_path, self.task, self.partition, shuffle = self.shuffle)

class DatasetProcessor():
    def __init__(self, root_folder, config, train = True):
        assert config["files_per_batch"] % config["num_workers"] == 0
        assert np.abs(np.sum(list(config["mask_token_prob"].values())) - 1) <= 1e-8

        self.config = copy.deepcopy(config)
        self.config["root_folder"] = root_folder

        if train:
            with open(os.path.join(root_folder, "train_files.json"), "r") as f:
                files = json.load(f)
            self.files = [os.path.join(root_folder, file) for file in files]
            print(f"Number of Training Files: {len(self.files)}")
        else:
            with open(os.path.join(root_folder, "val_files.json"), "r") as f:
                files = json.load(f)
            self.files = [os.path.join(root_folder, file) for file in files]
            print(f"Number of Validation Files: {len(self.files)}")

        tokenizer = self.get_tokenizer()

        cls_token = tokenizer.convert_tokens_to_ids(["[CLS]"])
        assert len(cls_token) == 1
        cls_token = cls_token[0]

        sep_token = tokenizer.convert_tokens_to_ids(["[SEP]"])
        assert len(sep_token) == 1
        sep_token = sep_token[0]

        mask_token = tokenizer.convert_tokens_to_ids(["[MASK]"])
        assert len(mask_token) == 1
        mask_token = mask_token[0]

        vocab_size = config["vocab_size"]

        self.vocab_size = vocab_size
        self.config["num_files"] = len(self.files)
        self.config["mask_token"] = mask_token
        self.config["sep_token"] = sep_token
        self.config["cls_token"] = cls_token
        self.config["vocab_size"] = vocab_size
        self.config["combine_doc"] = config["max_seq_len"] > 512

        self.cache = {}

        print(self.config)

    def get_tokenizer(self):
        return tokenization.FullTokenizer(
            vocab_file = os.path.join(self.config["root_folder"], "30k-corpus-uncased.vocab"),
            do_lower_case = True,
            spm_model_file = os.path.join(self.config["root_folder"], "30k-corpus-uncased.model")
        )

    def sample_mask_token(self, token):
        if token in [self.config["cls_token"], self.config["sep_token"]]:
            return token
        r = random.random()
        if r < self.config["mask_token_prob"]["mask"]:
            return self.config["mask_token"]
        elif r < self.config["mask_token_prob"]["mask"] + self.config["mask_token_prob"]["original"]:
            return token
        else:
            return random.randrange(start = 1, stop = self.config["vocab_size"])

    def get_masked_sentence(self, sentence, segment_ids):
        candidates = []
        for idx in range(len(sentence)):
            if sentence[idx] not in [self.config["sep_token"], self.config["cls_token"]]:
                candidates.append(idx)

        target_mask_size = min(self.config["max_mask_token"], int(len(candidates) * self.config["max_mask_ratio"]))

        mask_pos_ids = np.random.choice(candidates, size = target_mask_size, replace = False, p = None)
        mask_pos_ids = sorted(mask_pos_ids.tolist())

        masked_sentence = copy.deepcopy(sentence)
        mask_token = []
        mask_segment_ids = []
        mask_label = []
        for pos_ids in mask_pos_ids:
            sampled_mask = self.sample_mask_token(sentence[pos_ids])
            masked_sentence[pos_ids] = sampled_mask
            mask_token.append(sampled_mask)
            mask_segment_ids.append(segment_ids[pos_ids])
            mask_label.append(sentence[pos_ids])

        return masked_sentence, mask_token, mask_pos_ids, mask_segment_ids, mask_label

    def process_instance(self, instance, mask_sentence = True, include_label = True):

        sentence = instance["tokens"]
        segment_ids = instance["segment_ids"]

        pos_ids = list(range(len(sentence)))
        sentence_mask = [1] * len(sentence)

        masked_result = self.get_masked_sentence(sentence, segment_ids)
        masked_sentence, mask_token, mask_pos_ids, mask_segment_ids, mask_label = masked_result
        label_mask = [1] * len(mask_token)

        assert len(masked_sentence) == len(sentence)
        assert len(mask_token) <= self.config["max_mask_token"]
        assert len(mask_token) == len(mask_pos_ids)
        assert len(mask_token) == len(mask_segment_ids)
        assert len(mask_token) == len(mask_label)
        assert len(mask_token) == len(label_mask)

        zero_padding = [0] * (self.config["max_seq_len"] - len(sentence))
        sentence = np.asarray(sentence + zero_padding, np.int32)
        pos_ids = np.asarray(pos_ids + zero_padding, np.int32)
        segment_ids = np.asarray(segment_ids + zero_padding, np.int32)
        sentence_mask = np.asarray(sentence_mask + zero_padding, np.int32)
        masked_sentence = np.asarray(masked_sentence + zero_padding, np.int32)

        zero_padding = [0] * (self.config["max_mask_token"] - len(mask_token))
        mask_token = np.asarray(mask_token + zero_padding, np.int32)
        mask_pos_ids = np.asarray(mask_pos_ids + zero_padding, np.int32)
        mask_segment_ids = np.asarray(mask_segment_ids + zero_padding, np.int32)
        mask_label = np.asarray(mask_label + zero_padding, np.int32)
        label_mask = np.asarray(label_mask + zero_padding, np.int32)

        inst = {
            "sentence":sentence,
            "pos_ids":pos_ids, "segment_ids":segment_ids, "sentence_mask":sentence_mask,
            "mask_token":mask_token, "mask_pos_ids":mask_pos_ids, "mask_segment_ids":mask_segment_ids,
            "mask_label":mask_label, "label_mask":label_mask
        }

        if mask_sentence:
            inst["masked_sentence"] = masked_sentence
        else:
            inst["masked_sentence"] = sentence

        if include_label:
            assert instance["sentence_label"] is not None
            if type(instance["sentence_label"]) == bool:
                inst["sentence_label"] = 1 if instance["sentence_label"] else 0
            else:
                inst["sentence_label"] = instance["sentence_label"]

        return inst

    def process_file(args):
        processor, files = args

        max_seq_length = processor.config["max_seq_len"]
        short_seq_prob = processor.config["short_seq_prob"]
        cls_token = processor.config["cls_token"]
        sep_token = processor.config["sep_token"]

        instances = []
        for file_idx in range(len(files)):
            file = files[file_idx]
            try:
                with open(file, 'rb') as f:
                    documents = pickle.load(f)
            except:
                continue
            if processor.config["combine_doc"]:
                big_documents = []
                for document in documents:
                    big_documents.extend(document)
                raw_instances = processor.create_instances_from_document(big_documents)
                for raw_instance in raw_instances:
                    if random.random() < processor.config["drop_inst_prob"]:
                        continue
                    instances.append(processor.process_instance(raw_instance))
            else:
                for document in documents:
                    raw_instances = processor.create_instances_from_document(document)
                    for raw_instance in raw_instances:
                        if random.random() < processor.config["drop_inst_prob"]:
                            continue
                        instances.append(processor.process_instance(raw_instance))

        return instances

    def pretrain_task_generator(self):
        async_result = None
        async_pool = None

        while True:

            try:
                instances = []

                t0 = time.time()
                if self.config["num_workers"] <= 1:
                    file_batch = np.random.choice(self.files, size = self.config["files_per_batch"], replace = False)
                    instances = DatasetProcessor.process_file((self, file_batch))
                else:
                    file_batch = np.random.choice(self.files, size = self.config["files_per_batch"], replace = False)
                    args = [(self, []) for _ in range(self.config["num_workers"])]
                    for file_idx in range(len(file_batch)):
                        args[file_idx % self.config["num_workers"]][1].append(file_batch[file_idx])

                    if async_pool is not None:
                        async_result.ready()
                        returned_insts_list = async_result.get()
                        for returned_insts in returned_insts_list:
                            instances.extend(returned_insts)
                        async_pool.close()
                        async_pool.join()
                        async_pool = None
                        async_result = None

                    if async_pool is None:
                        async_pool = Pool(self.config["num_workers"])
                        async_result = async_pool.map_async(DatasetProcessor.process_file, args)

                t1 = time.time()
                retrieve_time = round(t1 - t0, 2)

                t0 = time.time()
                random.shuffle(instances)
                t1 = time.time()
                shuffle_time = round(t1 - t0, 2)

                print(f"Gen {len(instances)} insts. Retrieve Time: {retrieve_time} Shuffle Time: {shuffle_time}")

                for inst in instances:
                    yield inst

                instances = None
                gc.collect()
            except Exception as e:
                if async_pool is not None:
                    async_pool.terminate()
                    async_pool.close()
                raise e

    def downsteam_task_generator(self, file_path, task, partition, shuffle = True):

        max_seq_length = self.config["max_seq_len"]
        cls_token = self.config["cls_token"]
        sep_token = self.config["sep_token"]

        if f"{file_path}/{task}/{partition}" in self.cache:
            raw_instances = self.cache[f"{file_path}/{task}/{partition}"]
        else:
            raw_instances = self.create_instances_from_downsteam_task_file(file_path, task, partition)
            self.cache[f"{file_path}/{task}/{partition}"] = raw_instances

        if shuffle:
            random.shuffle(raw_instances)

        for raw_instance in raw_instances:
            yield self.process_instance(
                raw_instance,
                mask_sentence = False,
                include_label = raw_instance["sentence_label"] is not None)


    def create_instances_from_document(self, raw_document):
        def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_num_tokens:
                    break
                trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                assert len(trunc_tokens) >= 1
                if random.random() < 0.5:
                    del trunc_tokens[0]
                else:
                    trunc_tokens.pop()

        max_seq_length = self.config["max_seq_len"]
        short_seq_prob = self.config["short_seq_prob"]
        cls_token = self.config["cls_token"]
        sep_token = self.config["sep_token"]

        max_num_tokens = max_seq_length - 3
        instances = []
        current_chunk = []
        current_length = 0
        if random.random() < short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        else:
            target_seq_length = max_num_tokens

        document = []
        for segment in raw_document:
            if len(segment) == 0:
                print("detected empty segment")
                continue
            document.append(segment)

        for i in range(len(document)):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                is_swap_order = False
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                if len(current_chunk) == 1:
                    is_swap_order = True
                    target_b_length = target_seq_length - len(tokens_a)

                    random_start = random.randint(0, len(document) - 1)
                    for j in range(random_start, len(document)):
                        tokens_b.extend(document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                else:
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                if random.random() < 0.5:
                    is_swap_order = True
                    tokens_a, tokens_b = tokens_b, tokens_a

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = [cls_token] + tokens_a + [sep_token] + tokens_b + [sep_token]
                segment_ids = [0] + [0] * len(tokens_a) + [0] + [1] * len(tokens_b) + [1]
                instances.append({"tokens":tokens, "segment_ids":segment_ids, "sentence_label":is_swap_order})

                current_chunk = []
                current_length = 0
                if random.random() < short_seq_prob:
                    target_seq_length = random.randint(2, max_num_tokens)
                else:
                    target_seq_length = max_num_tokens

        return instances

    def create_instances_from_downsteam_task_file(self, file, task, partition):
        tokenizer = self.get_tokenizer()
        max_seq_length = self.config["max_seq_len"]
        cls_token = self.config["cls_token"]
        sep_token = self.config["sep_token"]

        max_num_tokens = max_seq_length - 3

        with open(file, "rb") as f:
            data = pickle.load(f)[task]["data"]

        instances = []

        print(f"{partition} number of instances: {len(data[partition])}")

        for inst in data[partition]:
            sentence_label = inst["label"]
            tokens_a = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inst["sentence_0"].lower()))
            if inst["sentence_1"] is None:
                tokens_b = []
            else:
                tokens_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inst["sentence_1"].lower()))

            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_num_tokens:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

            assert len(tokens_a) >= 1

            if len(tokens_b) == 0:
                tokens = [cls_token] + tokens_a + [sep_token]
                segment_ids = [0] + [0] * len(tokens_a) + [0]
            else:
                tokens = [cls_token] + tokens_a + [sep_token] + tokens_b + [sep_token]
                segment_ids = [0] + [0] * len(tokens_a) + [0] + [1] * len(tokens_b) + [1]

            instances.append({"tokens":tokens, "segment_ids":segment_ids, "sentence_label":sentence_label})

        return instances
