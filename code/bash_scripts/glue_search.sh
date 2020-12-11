
python3 /code/compile_model.py

CUDA_VISIBLE_DEVICES=0 python3 /code/run_glue.py --batch_size 32 --lr 2e-5 --epoch 5 $* &
CUDA_VISIBLE_DEVICES=1 python3 /code/run_glue.py --batch_size 32 --lr 3e-5 --epoch 5 $* &
CUDA_VISIBLE_DEVICES=2 python3 /code/run_glue.py --batch_size 32 --lr 4e-5 --epoch 5 $* &
CUDA_VISIBLE_DEVICES=3 python3 /code/run_glue.py --batch_size 32 --lr 5e-5 --epoch 5 $* &
CUDA_VISIBLE_DEVICES=4 python3 /code/run_glue.py --batch_size 16 --lr 2e-5 --epoch 5 $* &
CUDA_VISIBLE_DEVICES=5 python3 /code/run_glue.py --batch_size 16 --lr 3e-5 --epoch 5 $* &
CUDA_VISIBLE_DEVICES=6 python3 /code/run_glue.py --batch_size 16 --lr 4e-5 --epoch 5 $* &
CUDA_VISIBLE_DEVICES=7 python3 /code/run_glue.py --batch_size 16 --lr 5e-5 --epoch 5 $* &
wait
