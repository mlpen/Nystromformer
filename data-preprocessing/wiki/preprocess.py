import json
import pickle
import re
import os

if not os.path.exists("data"):
    os.mkdir("data")

def read_file(path):
    f = open(path, encoding = 'utf-8')
    text = f.read()
    f.close()
    docs = text.split("</doc>")
    assert docs[-1].strip() == ""
    result = []
    for doc in docs[:-1]:
        segments = doc.split("\n")
        start_idx = 0
        while not segments[start_idx].strip().startswith("<doc id"):
            start_idx += 1
            assert start_idx + 2 < len(segments)
        assert segments[start_idx + 1].strip() != ""
        assert segments[start_idx + 2].strip() == ""

        article = []
        for seg in segments[start_idx + 2:]:
            seg = seg.strip()
            if len(seg) == 0:
                continue
            article.append(seg)

        result.append(article)
    return result

root = "english_wiki"
folders = [os.path.join(root, folder) for folder in os.listdir(root)]
folders = sorted(folders)

max_num_tokens = 20000000
docs = []
num_tokens = 0
total_count = 0
file_idx = 0

for i in range(len(folders)):
    folder = folders[i]
    print(i, len(folders), folder)
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.startswith('wiki')]
    files = sorted(files)
    for j in range(len(files)):
        file = files[j]
        print(j, len(files), file)
        articles = read_file(file)
        for article in articles:

            text = "\n".join(article)
            approx_len = len(re.split(" |\n", text))

            if approx_len >= 128:
                docs.append(text)
                num_tokens += approx_len

                if num_tokens > max_num_tokens:
                    with open(f"data/english_wiki-partition-{file_idx:04}.txt", "w", encoding = "utf-8") as dump_f:
                        for doc in docs:
                            dump_f.write(doc)
                            dump_f.write("\n\n")
                    print(f"dumped data/english_wiki-partition-{file_idx:04}.txt")

                    file_idx += 1
                    docs = []
                    num_tokens = 0

            total_count += 1
            if total_count % 10000 == 0:
                print(f"{total_count}, {len(docs)}, {num_tokens}, {num_tokens / max_num_tokens}")

with open(f"data/english_wiki-partition-{file_idx:04}.txt", "w", encoding = "utf-8") as dump_f:
    for doc in docs:
        dump_f.write(doc)
        dump_f.write("\n\n")
print(f"dumped data/english_wiki-partition-{file_idx:04}.txt")


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
    file = f"data/english_wiki-partition-{file_idx:04}.txt"
    args.append((file, file.replace(".txt", "-roberta-base.pickle")))

    if len(args) == num_workers:
        if num_workers == 1:
            tokenize(args[0])
        else:
            pool.map(tokenize, args)
        args = []
if num_workers > 1:
    pool.close()
