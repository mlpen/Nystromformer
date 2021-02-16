import sys
sys.path.append("./long-range-arena/lra_benchmarks/image/")
import input_pipeline
import numpy as np
import pickle

train_ds, eval_ds, test_ds, num_classes, vocab_size, input_shape = input_pipeline.get_cifar10_datasets(
    n_devices = 1, batch_size = 1, normalize = False)

mapping = {"train":train_ds, "dev": eval_ds, "test":test_ds}
for component in mapping:
    ds_list = []
    for idx, inst in enumerate(iter(mapping[component])):
        ds_list.append({
            "input_ids_0":inst["inputs"].numpy()[0].reshape(-1),
            "label":inst["targets"].numpy()[0]
        })
        if idx % 100 == 0:
            print(f"{idx}\t\t", end = "\r")
    with open(f"image.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)
