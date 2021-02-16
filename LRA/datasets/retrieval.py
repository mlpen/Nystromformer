import sys
sys.path.append("./long-range-arena/lra_benchmarks/matching/")
import input_pipeline
import numpy as np
import pickle

train_ds, eval_ds, test_ds, encoder = input_pipeline.get_matching_datasets(
    n_devices = 1, task_name = None, data_dir = "./lra_release/lra_release/tsv_data/",
    batch_size = 1, fixed_vocab = None, max_length = 4000, tokenizer = "char",
    vocab_file_path = None)

mapping = {"train":train_ds, "dev": eval_ds, "test":test_ds}
for component in mapping:
    ds_list = []
    for idx, inst in enumerate(iter(mapping[component])):
        ds_list.append({
            "input_ids_0":np.concatenate([inst["inputs1"].numpy()[0], np.zeros(96, dtype = np.int32)]),
            "input_ids_1":np.concatenate([inst["inputs2"].numpy()[0], np.zeros(96, dtype = np.int32)]),
            "label":inst["targets"].numpy()[0]
        })
        if idx % 100 == 0:
            print(f"{idx}\t\t", end = "\r")
    with open(f"retrieval.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)
