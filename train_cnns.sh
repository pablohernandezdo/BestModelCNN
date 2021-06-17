#!/bin/bash

# CNN1
echo "Training model CNN_6k_6k, lr = 1e-3, epochs = 20, batch_size = 256"
python train.py \
        --lr 1e-3 \
        --epochs 20 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 1 \
        --model_folder 'models'  \
        --classifier CNN1 \
        --model_name CNN1_1e3_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/Train_constant.npy" \
        --val_path "Data/TrainReady/Val_constant.npy"
