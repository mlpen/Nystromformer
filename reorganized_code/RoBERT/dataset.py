import torch
import torch.nn as nn
import math
from torch.utils.data.dataset import Dataset
import sys
import os
import random
import json
import pickle
import numpy as np
from multiprocessing import Pool

class CorpusDataset(Dataset):
    def __init__(self, folder_path, file_json, option = None):

        with open(os.path.join(folder_path, file_json), "r") as f:
            files = json.load(f)

        self.files = [os.path.join(folder_path, file) for file in files]
        if option is not None:
            if "bert_dataset" in option and option["bert_dataset"]:
                self.files = [file for file in self.files if "bookcorpus" in file or "english_wiki" in file]
            self.load_all_insts = option["load_all_insts"] if "load_all_insts" in option else False
            self.files_per_batch = option["files_per_batch"] if "files_per_batch" in option else 1024
            self.keep_prob = option["keep_prob"] if "keep_prob" in option else 0.1

        print(f"Number of Files: {len(self.files)}", flush = True)

        self.curr_idx = 0
        self.examples = []

    def load_files(self):

        if self.load_all_insts:
            if len(self.examples) == 0:
                for idx, file in enumerate(self.files):
                    print(f"Loading {idx} / {len(self.files)}: {file}", flush = True)
                    with open(file, "rb") as f:
                        self.examples.extend(pickle.load(f))

            self.curr_idx = 0
            random.shuffle(self.examples)
            print(f"Number of Instances: {len(self.examples)}", flush = True)
            print(f"Completed Loading", flush = True)
        else:
            del self.examples

            selected_files = np.random.choice(self.files, size = min(len(self.files), self.files_per_batch), replace = False)
            self.curr_idx = 0
            self.examples = []
            for idx, file in enumerate(selected_files):
                print(f"Loading {idx} / {len(selected_files)}: {file}", flush = True)
                with open(file, "rb") as f:
                    self.examples.extend([inst for inst in pickle.load(f) if random.random() < self.keep_prob])

            random.shuffle(self.examples)
            print(f"Number of Instances: {len(self.examples)}", flush = True)
            print(f"Completed Loading", flush = True)


    def __len__(self):
        return 100000000

    def __getitem__(self, i) -> torch.Tensor:
        if self.curr_idx >= len(self.examples):
            self.load_files()
        inst = self.examples[self.curr_idx]
        self.curr_idx += 1
        return torch.tensor(inst, dtype = torch.long)


class CorpusDatasetV2(Dataset):
    def __init__(self, folder_path, file_json, option = None):

        with open(os.path.join(folder_path, file_json), "r") as f:
            files = json.load(f)

        self.files = [os.path.join(folder_path, file) for file in files]
        if option is not None and "bert_dataset" in option and option["bert_dataset"]:
            self.files = [file for file in self.files if "bookcorpus" in file or "english_wiki" in file]

        print(f"Number of Files: {len(self.files)}", flush = True)

        self.examples = []
        for idx, file in enumerate(self.files):
            print(f"Loading {idx} / {len(self.files)}: {file}", flush = True)
            with open(file, "rb") as f:
                self.examples.extend(pickle.load(f))

        random.shuffle(self.examples)
        print(f"Number of Instances: {len(self.examples)}", flush = True)
        print(f"Completed Loading", flush = True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype = torch.long)

class WiKiHopDataset(Dataset):
    def __init__(self, tokenizer, folder_path, file, num_workers = 8):

        self.tokenizer = tokenizer
        self.max_seq_len = self.tokenizer.model_max_length

        dump_file = f"/model/wikihop-{file}-{self.max_seq_len}.pickle"

        if os.path.exists(dump_file):
            with open(dump_file, "rb") as f:
                self.examples = pickle.load(f)
        else:
            args = [(idx, self, []) for idx in range(num_workers)]
            with open(os.path.join(folder_path, file), "r") as f:
                wikihop = json.load(f)
            for idx, inst in enumerate(wikihop):
                inst["idx"] = idx
                args[idx % num_workers][-1].append(inst)

            with Pool(num_workers) as pool:
                gathered_results = pool.map(WiKiHopDataset.process_batches, args)

            self.examples = []
            for results in gathered_results:
                self.examples.extend(results)

            with open(dump_file, "wb") as f:
                pickle.dump(self.examples, f)

        print(f"Number of Instances: {self.__len__()}", flush = True)

    def process_batches(args):
        worker_idx, processor, batches = args
        results = []
        for idx, inst in enumerate(batches):
            results.extend(processor.process_instance(inst))
            if idx % 100 == 0:
                print(f"Worker {worker_idx}, Preprocessing: {idx} / {len(batches)}, Got {len(results)} instances", flush = True)
        return results

    def process_instance(self, inst):

        def tokenize(string):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(string))

        inst_id = inst["id"]
        inst_idx = inst["idx"]
        supports = inst["supports"]
        query = inst["query"]
        candidates = inst["candidates"]
        answer = inst["answer"]
        assert answer in candidates

        candidates_pos = []
        prefix = tokenize("<question>") + tokenize(query) + tokenize("</question>")

        for candidate in candidates:
            candidates_pos.append(len(prefix))
            prefix.extend(tokenize("<answer>") + tokenize(candidate) + tokenize("</answer>"))

        answer_pos = candidates_pos[candidates.index(answer)]
        answer_token = tokenize("<answer>")
        assert len(answer_token) == 1
        for pos in candidates_pos:
            assert prefix[pos] == answer_token[0]

        prefix.extend(tokenize("</s>"))

        results = []

        pending_instance = prefix.copy()
        for support in supports:
            chunk = tokenize(support) + tokenize("</s>")
            if len(pending_instance) + len(chunk) > self.max_seq_len:
                results.append({
                    "id":inst_id, "idx":inst_idx,
                    "input_ids":pending_instance, "candidates":candidates_pos,
                    "answer":answer_pos})
                pending_instance = prefix.copy()
            else:
                pending_instance.extend(chunk)

        if len(pending_instance) > len(prefix):
            results.append({
                "id":inst_id, "idx":inst_idx,
                "input_ids":pending_instance, "candidates":candidates_pos,
                "answer":answer_pos})

        return results

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        input_ids = example["input_ids"] + [1] * (self.max_seq_len - len(example["input_ids"]))
        attention_mask = [1] * len(example["input_ids"]) + [0] * (self.max_seq_len - len(example["input_ids"]))
        candidate_mask = [0] * self.max_seq_len
        for candidate in example["candidates"]:
            candidate_mask[candidate] = 1

        return {
            "idx":torch.tensor(example["idx"], dtype = torch.long),
            "input_ids":torch.tensor(input_ids, dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "candidate_mask":torch.tensor(candidate_mask, dtype = torch.long),
            "answer_positions":torch.tensor(example["answer"], dtype = torch.long)
        }
