#!/bin/bash
set -e

# cremad - distill
CUDA_VISIBLE_DEVICES=3 python train_CRE_mix_alter.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_D --name CRE_3 --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

# cremad - distill
CUDA_VISIBLE_DEVICES=3 python train_CRE_D.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_D --name CRE_softlabel --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

# cremad - distill
CUDA_VISIBLE_DEVICES=3 python train_CRE_D3.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_D3 --name CRE_4 --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

# cremad - distill
CUDA_VISIBLE_DEVICES=3 python train_CRE_latefusion.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_fusion --name CRE_latefusion --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100
