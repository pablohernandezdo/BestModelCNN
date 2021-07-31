#!/bin/bash

# CNN1
echo "Training model CNN1, lr = 1e-3, epochs = 200, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --epochs 200 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN1 \
        --model_name CNN1_1e3_256_fsc_200epch \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"