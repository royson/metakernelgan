#!/bin/bash
datapath='/datasets/super_res'
model='KernelGANFKP'
BM=('Set14' 'B100' 'Urban100' 'DIV2K')

# clear x2
for r in ${!BM[@]}; do
    python prepare_dataset.py --dataset_path $datapath --model $model --sf 2 --dataset ${BM[r]} --postfix _clean
done

# kernel noise 
for r in ${!BM[@]}; do
    python prepare_dataset.py --dataset_path $datapath --model $model --sf 2 --dataset ${BM[r]} --noise_ker 0.4 --postfix _noise_ker
done

# image noise 
for r in ${!BM[@]}; do
    python prepare_dataset.py --dataset_path $datapath --model $model --sf 2 --dataset ${BM[r]} --noise_im 0.039 --postfix _noise_img
done

# clear x4
for r in ${!BM[@]}; do
    python prepare_dataset.py --dataset_path $datapath --model $model --sf 4 --dataset ${BM[r]} --postfix _clean
done