import sys
sys.path.append("./long-range-arena/lra_benchmarks/listops/")
import input_pipeline
import numpy as np
import pickle

train_ds, eval_ds, test_ds, encoder = input_pipeline.get_datasets(
    n_devices = 1, task_name = "basic", data_dir = "./lra_release/lra_release/listops-1000/",
    batch_size = 1, max_length = 2000)

mapping = {"train":train_ds, "dev": eval_ds, "test":test_ds}
for component in mapping:
    ds_list = []
    for idx, inst in enumerate(iter(mapping[component])):
        ds_list.append({
            "input_ids_0":np.concatenate([inst["inputs"].numpy()[0], np.zeros(48, dtype = np.int32)]),
            "label":inst["targets"].numpy()[0]
        })
        if idx % 100 == 0:
            print(f"{idx}\t\t", end = "\r")
    with open(f"listops.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)
