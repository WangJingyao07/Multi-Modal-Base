#!/bin/bash
set -e

# cremad - base
CUDA_VISIBLE_DEVICES=1 python train_CRE.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE --name CRE_librosa --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0

# cremad - base
CUDA_VISIBLE_DEVICES=3 python train_CRE_OGM.py --batch_size 64 --gradient_accumulation_steps 40 --savedir ./saved/CRE_oge --name CRE_10 --task CREMAD --task_type classification --model mml_av --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

# cremad - base
CUDA_VISIBLE_DEVICES=3 python train_CRE_MLU.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_MLU --name CRE_no_sgd --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

# cremad - base
CUDA_VISIBLE_DEVICES=1 python train_CRE_mix.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_mix --name CRE_new—sgd --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

