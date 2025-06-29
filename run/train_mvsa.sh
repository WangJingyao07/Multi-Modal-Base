#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python ../train/train_mvsa.py --batch_sz 16 --gradient_accumulation_steps 40 --savedir ./saved/MVSA_VAE --name MVSA_Single_vaenokl --task MVSA_Single --task_type classification --model mml_vt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1 --noise 0.0 --dataset MVSA
