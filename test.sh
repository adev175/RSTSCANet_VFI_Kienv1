#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py \
--datasetName Vimeo_90K \
--datasetPath 'data\vimeo_triplet' \
--checkpoint_dir 'checkpoints\rstscanet.pth' \
--save_path 'checkpoints\try1'
