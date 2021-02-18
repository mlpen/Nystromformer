
## LRA Benchmark

We released the source code for LRA benchmark.

To prepare the datasets, one would need
```
tensorboard>=2.3.0, tensorflow>=2.3.1, tensorflow-datasets>=4.0.1
```

To prepare the datasets, one would need to download the source code from [LRA repo](https://github.com/google-research/long-range-arena) and place `long-range-arena` folder in folder `LRA/datasets/` and also download [lra_release.gz](https://storage.googleapis.com/long-range-arena/lra_release.gz) released by LRA repo and place the unzipped folder in folder `LRA/datasets/`. The directory structure would be
```
LRA/datasets/long-range-arena
LRA/datasets/lra_release
```
Then, run `sh create_datasets.sh` and it will create train, dev, and test dataset pickle files for each task.

To run the LRA tasks, one would need
```
pytorch==1.7.1, transformers==3.3.1, performer-pytorch
```
To run a LRA experiment, run the following command in `code` folder
```
python3 run_tasks.py --model <model> --task <task>
```
where `<model>` can be set to `softmax, nystrom-64, reformer-2, performer-256` corresponding to standard self-attention, Nystromformer with 64 landmarks, Reformer with 2 LSHs, Performer with 256 random projection dimension. And `<task>` can be set to `listops, text, retrieval, image, pathfinder32-curv_contour_length_14`. The best models and log files will be saved `LRA/logs/` folder.
