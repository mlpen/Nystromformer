
## Nystromformer for Image Classification

We used Nystromformer for image classification on ImageNet. Our codes are based on [T2T-ViT](https://github.com/yitu-opensource/T2T-ViT).

### 1. Download T2T-ViT codebase. Then add token_nystromformer.py in the model subfolder and replace t2t_vit.py with our t2t_vit.py.

### 2. Requirements
Follow T2T-ViT instruction to install all the required packages and prepare ImageNet. 

### 3. Deploy T2T-ViT_t-14 with Nystromformer without retraining.
Download T2T-ViT_t-14 at [T2T-ViT](https://github.com/yitu-opensource/T2T-ViT), then deploy it by running:
```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model T2t_vit_nys_14 -b 100 --eval_checkpoint path/to/checkpoint
```

### 4. Train T2T-ViT with Nystromformer.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 path/to/data --model T2t_vit_nys_7 -b 128 --lr 1e-3 --weight-decay .03 --amp --img-size 224
```