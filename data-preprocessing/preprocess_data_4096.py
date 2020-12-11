import json
import pickle
import os
import random
from multiprocessing import Pool
from transformers import RobertaTokenizerFast

dump_folder = "4096-roberta"
if not os.path.exists(dump_folder):
    os.mkdir(dump_folder)

def process(args):
    dataset, percentage = args

    data_folder = os.path.join(dataset, "data")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    max_seq_len = 4096
    per_batch_inst = 1024
    block_size = max_seq_len - tokenizer.num_special_tokens_to_add(pair = False)

    random.seed(hash(dataset))

    files = sorted([os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".pickle")])
    data_buffer = []

    batch = []
    file_idx = 0

    for file in files:
        print(file)
        with open(file, "rb") as f:
            data = pickle.load(f)
        data_buffer.extend(data)
        for start_idx in range(0, len(data_buffer) - block_size + 1, block_size):
            block = data_buffer[start_idx:(start_idx + block_size)]
            assert len(block) == block_size
            if random.random() < percentage:
                block = tokenizer.build_inputs_with_special_tokens(block)
                assert len(block) == max_seq_len
                batch.append(block)
            if len(batch) >= per_batch_inst:
                dump_path = os.path.join(dump_folder, f"{dataset}-{file_idx:05}.pickle")
                with open(dump_path, "wb") as dump_f:
                    pickle.dump(batch, dump_f)
                batch = []
                file_idx += 1
                print(dump_path)
        data_buffer = data_buffer[(start_idx + block_size):]

    dump_path = os.path.join(dump_folder, f"{dataset}-{file_idx:05}.pickle")
    with open(dump_path, "wb") as dump_f:
        pickle.dump(batch, dump_f)
    batch = []
    file_idx += 1
    print(dump_path)


datasets = [("bookcorpus", 1.0), ("english_wiki", 1.0), ("realnews", 1 / 3), ("stories", 1 / 3)]
pool = Pool(len(datasets))
pool.map(process, datasets)
pool.close()

import json
files = sorted([file for file in os.listdir(dump_folder) if file.endswith(".pickle")])
print(json.dumps(files, indent = 4))
random.seed(1)
random.shuffle(files)
split = int(len(files) * 0.1)

with open(os.path.join(dump_folder, "dev.json"), "w") as f:
    json.dump(files[:split], f, indent = 4)

with open(os.path.join(dump_folder, "train.json"), "w") as f:
    json.dump(files[split:], f, indent = 4)
