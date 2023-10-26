#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python main.py \
python main.py \
--datasetName Vimeo_90K \
--datasetPath 'data/vimeo_triplet' \
--batch_size 1 \
--checkpoint_dir 'checkpoints\rstscanet.pth' \
--max_epoch 200 \
--resume True\
