import json
import pickle
import re
import os

if not os.path.exists("data"):
    os.mkdir("data")

files = os.listdir("stories")

max_num_tokens = 20000000
docs = []
num_tokens = 0
total_count = 0
file_idx = 0
for file in files:
    file = os.path.join("stories", file)

    try:

        with open(file, "r", encoding = "utf-8") as read_f:

            text = read_f.read()
            approx_len = len(re.split(" |\n", text))

            docs.append(text)
            num_tokens += approx_len

            if num_tokens > max_num_tokens:
                with open(f"data/stories-partition-{file_idx:04}.txt", "w", encoding = "utf-8") as dump_f:
                    for doc in docs:
                        dump_f.write(doc)
                        dump_f.write("\n\n")
                print(f"dumped data/stories-partition-{file_idx:04}.txt")

                file_idx += 1
                docs = []
                num_tokens = 0

    except Exception as e:
        print(e)

    total_count += 1
    print(f"{total_count}/{len(files)}, {len(docs)}, {num_tokens}, {num_tokens / max_num_tokens}")

with open(f"data/stories-partition-{file_idx:04}.txt", "w", encoding = "utf-8") as dump_f:
    for doc in docs:
        dump_f.write(doc)
        dump_f.write("\n\n")
print(f"dumped data/stories-partition-{file_idx:04}.txt")


import pickle
from multiprocessing import Pool
import os
from transformers import RobertaTokenizerFast

def tokenize(args):
    src, tgt = args

    if not os.path.exists(src):
        return

#     print(src, tgt)
#     return

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    print("START", flush = True)
    with open(src, "r", encoding = "utf-8") as read_f:
        text = read_f.read()
    print(f"Read {src}", flush = True)

    tokens = tokenizer.tokenize(text)
    print(f"Tokenized {src}", flush = True)
    del text

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"To Token IDs {src}", flush = True)

    with open(tgt, "wb") as dump_f:
        pickle.dump(token_ids, dump_f)
    print(f"Dump {tgt}", flush = True)
    print("END", flush = True)

num_workers = 2
if num_workers > 1:
    pool = Pool(num_workers)
args = []
for file_idx in range(10000):
    file = f"data/stories-partition-{file_idx:04}.txt"
    args.append((file, file.replace(".txt", "-roberta-base.pickle")))

    if len(args) == num_workers:
        if num_workers == 1:
            tokenize(args[0])
        else:
            pool.map(tokenize, args)
        args = []
if num_workers > 1:
    pool.close()
