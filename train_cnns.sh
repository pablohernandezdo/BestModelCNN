#!/bin/bash

# CNN_6K_6K
#echo "Training model CNN_6k_6k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_6k \
#        --model_name CNN_6k_6k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P1=$!
#
#echo "Training model CNN_6k_6k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_6k \
#        --model_name CNN_6k_6k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_6k_6k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_6k \
#        --model_name CNN_6k_6k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
## CNN_6K_5K
#echo "Training model CNN_6k_5k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_5k \
#        --model_name CNN_6k_5k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_6k_5k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_5k \
#        --model_name CNN_6k_5k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_6k_5k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_5k \
#        --model_name CNN_6k_5k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
## CNN_6K_4K
#echo "Training model CNN_6k_4k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_4k \
#        --model_name CNN_6k_4k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_6k_4k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_4k \
#        --model_name CNN_6k_4k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_6k_4k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_4k \
#        --model_name CNN_6k_4k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
## CNN_6K_3K
#echo "Training model CNN_6k_3k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_3k \
#        --model_name CNN_6k_3k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_6k_3k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_3k \
#        --model_name CNN_6k_3k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_6k_3k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_3k \
#        --model_name CNN_6k_3k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
## CNN_6K_2K
#echo "Training model CNN_6k_2k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_2k \
#        --model_name CNN_6k_2k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_6k_2k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_2k \
#        --model_name CNN_6k_2k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_6k_2k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_2k \
#        --model_name CNN_6k_2k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
## CNN_6K_1K
#echo "Training model CNN_6k_1k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_1k \
#        --model_name CNN_6k_1k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_6k_1k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_1k \
#        --model_name CNN_6k_1k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_6k_1k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_6k_1k \
#        --model_name CNN_6k_1k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
## CNN_5K_5K
#echo "Training model CNN_5k_5k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_5k \
#        --model_name CNN_5k_5k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_5k_5k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_5k \
#        --model_name CNN_5k_5k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_5k_5k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_5k \
#        --model_name CNN_5k_5k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
## CNN_5K_4K
#echo "Training model CNN_5k_4k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_4k \
#        --model_name CNN_5k_4k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_5k_4k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_4k \
#        --model_name CNN_5k_4k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_5k_4k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_4k \
#        --model_name CNN_5k_4k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
## CNN_5K_3K
#echo "Training model CNN_5k_3k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_3k \
#        --model_name CNN_5k_3k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_5k_3k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_3k \
#        --model_name CNN_5k_3k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_5k_3k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_3k \
#        --model_name CNN_5k_3k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
## CNN_5K_2K
#echo "Training model CNN_5k_2k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_2k \
#        --model_name CNN_5k_2k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_5k_2k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_2k \
#        --model_name CNN_5k_2k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_5k_2k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_2k \
#        --model_name CNN_5k_2k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
## CNN_5K_1K
#echo "Training model CNN_5k_1k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_1k \
#        --model_name CNN_5k_1k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_5k_1k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_1k \
#        --model_name CNN_5k_1k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_5k_1k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_5k_1k \
#        --model_name CNN_5k_1k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
## CNN_4K_4K
#echo "Training model CNN_4k_4k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_4k_4k \
#        --model_name CNN_4k_4k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_4k_4k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_4k_4k \
#        --model_name CNN_4k_4k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_4k_4k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_4k_4k \
#        --model_name CNN_4k_4k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
## CNN_4K_3K
#echo "Training model CNN_4k_3k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_4k_3k \
#        --model_name CNN_4k_3k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_4k_3k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_4k_3k \
#        --model_name CNN_4k_3k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_4k_3k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_4k_3k \
#        --model_name CNN_4k_3k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
## CNN_4K_2K
#echo "Training model CNN_4k_2k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_4k_2k \
#        --model_name CNN_4k_2k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_4k_2k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_4k_2k \
#        --model_name CNN_4k_2k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_4k_2k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_4k_2k \
#        --model_name CNN_4k_2k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2

# CNN_4K_1K
echo "Training model CNN_4k_1k, lr = 1e-3, epochs = 20, batch_size = 256"
python train.py \
        --lr 1e-3 \
        --epochs 20 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 1 \
        --model_folder 'models'  \
        --classifier CNN_4k_1k \
        --model_name CNN_4k_1k_1e3_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/Train_constant.npy" \
        --val_path "Data/TrainReady/Val_constant.npy"

P1=$!

echo "Training model CNN_4k_1k, lr = 1e-4, epochs = 20, batch_size = 256"
python train.py \
        --lr 1e-4 \
        --epochs 20 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 1 \
        --model_folder 'models'  \
        --classifier CNN_4k_1k \
        --model_name CNN_4k_1k_1e4_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/Train_constant.npy" \
        --val_path "Data/TrainReady/Val_constant.npy"

P2=$!
wait $P1 $P2

#echo "Training model CNN_4k_1k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_4k_1k \
#        --model_name CNN_4k_1k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &

#P1=$!
#
## CNN_3K_3K
#echo "Training model CNN_3k_3k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_3k_3k \
#        --model_name CNN_3k_3k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_3k_3k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_3k_3k \
#        --model_name CNN_3k_3k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_3k_3k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_3k_3k \
#        --model_name CNN_3k_3k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
## CNN_3K_2K
#echo "Training model CNN_3k_2k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_3k_2k \
#        --model_name CNN_3k_2k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_3k_2k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_3k_2k \
#        --model_name CNN_3k_2k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_3k_2k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_3k_2k \
#        --model_name CNN_3k_2k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
## CNN_3K_1K
#echo "Training model CNN_3k_1k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_3k_1k \
#        --model_name CNN_3k_1k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_3k_1k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_3k_1k \
#        --model_name CNN_3k_1k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_3k_1k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_3k_1k \
#        --model_name CNN_3k_1k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
## CNN_2K_2K
#echo "Training model CNN_2k_2k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_2k_2k \
#        --model_name CNN_2k_2k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_2k_2k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_2k_2k \
#        --model_name CNN_2k_2k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_2k_2k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_2k_2k \
#        --model_name CNN_2k_2k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
## CNN_2K_1K
#echo "Training model CNN_2k_1k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_2k_1k \
#        --model_name CNN_2k_1k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_2k_1k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_2k_1k \
#        --model_name CNN_2k_1k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_2k_1k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_2k_1k \
#        --model_name CNN_2k_1k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
## CNN_1K_1K
#echo "Training model CNN_1k_1k, lr = 1e-3, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-3 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_1k_1k \
#        --model_name CNN_1k_1k_1e3_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy" &
#
#P1=$!
#
#echo "Training model CNN_1k_1k, lr = 1e-4, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-4 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_1k_1k \
#        --model_name CNN_1k_1k_1e4_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
#
#P2=$!
#wait $P1 $P2
#
#echo "Training model CNN_1k_1k, lr = 1e-5, epochs = 20, batch_size = 256"
#python train.py \
#        --lr 1e-5 \
#        --epochs 20 \
#        --batch_size 256 \
#        --earlystop 0 \
#        --eval_iter 1 \
#        --model_folder 'models'  \
#        --classifier CNN_1k_1k \
#        --model_name CNN_1k_1k_1e5_256 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/Train_constant.npy" \
#        --val_path "Data/TrainReady/Val_constant.npy"
