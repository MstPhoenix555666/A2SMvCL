#!/bin/bash

# * laptop
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name gcnclbert --dataset laptop --seed 3407 --bert_lr 1e-5 --num_epoch 80 --max_length 100 --cuda 0 


# * restaurant
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name gcnclbert --dataset restaurant --seed 3407 --bert_lr 2e-5 --num_epoch 80 --max_length 100 --cuda 0 
CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name gcnclbert --dataset restaurant --seed 3407 --bert_lr 2e-5 --num_epoch 30 --max_length 100 --cuda 0 --num_layers 1


# * twitter
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name gcnclbert --dataset twitter --seed 3407 --bert_lr 2e-5 --num_epoch 70 --max_length 100 --cuda 0 



# * RES15
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name gcnclbert --dataset res15 --seed 3407 --bert_lr 2e-5 --num_epoch 80 --max_length 100 --cuda 0 


# * RES16
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name gcnclbert --dataset res16 --seed 3407 --bert_lr 2e-5 --num_epoch 80 --max_length 100 --cuda 0 
