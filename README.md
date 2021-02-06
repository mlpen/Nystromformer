# Nystromformer: A Nystrom-based Algorithm for Approximating Self-Attention

Transformers have emerged as a powerful workhorse for a broad range of natural language processing tasks. A key component that drives the impressive performance of Transformers is their self-attention mechanism that identifies/encodes the influence or dependence of other tokens for each specific token. Its benefits notwithstanding, the quadratic complexity of self-attention on the input sequence length has limited its application to longer sequences – a topic being actively studied in the community. To address this limitation, we propose Nystromformer – a model that exhibits excellent scalability as a function of sequence length. Our idea is based on adapting the Nystrom method to approximate the standard self-attention with an efficient O(n) complexity.

## Requirements

```
docker, nvidia-docker
```

## Datasets

The pretraining dataset consists of English Wikipedia and BookCorpus. For pretraining on long sequence, we added one third Stories and one third Realnews. All downloaded data files should be placed in the corresponding folder under `data-preprocessing`. The original format of English Wikipedia dump is preprocessed using
[wikiextractor](https://github.com/attardi/wikiextractor), and the resulting files are placed in `data-preprocessing/wiki`. Then, run `data-preprocessing/<dataset>/preprocess.py` under each corresponding folder to generate data files of unified format. After preprocessing, run `data-preprocessing/preprocess_data_<length>.py` to generate pretraining data of specific sequence length.

## Pretraining

To start pretraining of a specific configuration: create a folder `<model>` (for example, `nystrom-512`) and write `<model>/config.json` to specify model and training configuration, then under `<model>` folder, run
```
docker run --rm --name=pretrain \
  --network=host --ipc=host --gpus all \
  -v "$PWD/../data-preprocessing/512-roberta:/dataset" \
  -v "$PWD/../code:/code" \
  -v "$PWD:/model" \
  -d mlpen/bert_env:0 \
  /bin/bash -c \
  "python3 /code/run_pretrain.py >> /model/pretrain.txt 2>&1"
```
All outputs will be redirected to `<model>/pretrain.txt`. The command will create a `<model>/model` folder holding all checkpoints and log file. The training can be stopped anytime by running `docker kill pretrain`, and can be resumed from the last checkpoint using the same command for starting pretraining.

## Pretraining from Different Model's Checkpoint

Copy a checkpoint (one of `.model` or `.cp` file) from `<diff_model>/model` folder to `<model>` folder and add a key-value pair in `<model>/config.json`: `"from_cp": "/model/<checkpoint_file>"`. One example is shown in `nystrom-4096/config.json`. This procedure also works for extending the max sequence length of a model (For example, use `nystrom-512` pretrained weights as initialization for `nystrom-4096`).

## GLUE

To finetune model on GLUE tasks, download GLUE datasets and place them under `glue` folder, then under folder `<model>`, run
```
docker run --rm --name=glue \
  --network=host --ipc=host --gpus all \
  -v "$PWD/../glue:/glue" \
  -v "$PWD/../code:/code" \
  -v "$PWD:/model" \
  -d mlpen/bert_env:0 \
  /bin/bash -c \
  "python3 /code/run_glue.py --batch_size 32 --lr 3e-5 --epoch 5 --task MRPC --checkpoint 99 >> /model/glue.txt 2>&1"
```
`batch_size`, `lr`, `epoch`, `task`, `checkpoint` can be changed to finetune on different task, different hyperparameters, or different checkpoints. All outputs will be redirected to `<model>/glue.txt`. The log file is located at `<model>/model` folder.

## WikiHop

To finetune model on WikiHop tasks, download WikiHop datasets and place the `train.json` and `dev.json` under `wikihop` folder, then under folder `<model>`, run
```
docker run --rm --name=wikihop \
  --network=host --ipc=host --gpus all \
  -v "$PWD/../wikihop:/wikihop" \
  -v "$PWD/../code:/code" \
  -v "$PWD:/model" \
  -d mlpen/bert_env:0 \
  /bin/bash -c \
  "python3 /code/run_wikihop.py --batch_size 32 --lr 3e-5 --epoch 15 --checkpoint 69 >> /model/wikihop.txt 2>&1"
```
`batch_size`, `lr`, `epoch`, `checkpoint` can be changed to finetune on different hyperparameters or different checkpoints. All outputs will be redirected to `<model>/wikihop.txt`. The log file is located at `<model>/model` folder. The training can be stopped anytime by running `docker kill wikihop`, and can be resumed from the last checkpoint using the same command for starting pretraining.

## Citation
```
@article{xiong2021nystromformer,
  title={Nystr{\"o}mformer: A Nystr{\"o}m-based Algorithm for Approximating Self-Attention},
  author={Xiong, Yunyang and Zeng, Zhanpeng and Chakraborty, Rudrasis and Tan, Mingxing and Fung, Glenn and Li, Yin and Singh, Vikas},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```
