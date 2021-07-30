#!/bin/bash

# CNN1
echo "Training model CNN1, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN1 \
        --model_name CNN1_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model CNN1, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN1 \
        --model_name CNN1_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model CNN1, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN1 \
        --model_name CNN1_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P3=$!
wait $P1 $P2 $P3

echo "Training model CNN2, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN2 \
        --model_name CNN2_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model CNN2, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN2 \
        --model_name CNN2_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model CNN2, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN2 \
        --model_name CNN2_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P3=$!
wait $P1 $P2 $P3

echo "Training model CNN3, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN3 \
        --model_name CNN3_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model CNN3, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN3 \
        --model_name CNN3_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model CNN3, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN3 \
        --model_name CNN3_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P3=$!
wait $P1 $P2 $P3