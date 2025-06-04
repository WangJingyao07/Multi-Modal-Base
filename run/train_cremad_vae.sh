#!/bin/bash
set -e

# cremad - vae
CUDA_VISIBLE_DEVICES=2 python train_CRE_vae.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_vae --name CRE_ttt --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100
