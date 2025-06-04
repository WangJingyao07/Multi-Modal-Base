#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_CRE.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE --name CRE_librosa --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0

CUDA_VISIBLE_DEVICES=3 python train_CRE_OGM.py --batch_size 64 --gradient_accumulation_steps 40 --savedir ./saved/CRE_oge --name CRE_10 --task CREMAD --task_type classification --model mml_av --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=3 python train_CRE_MLU.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_MLU --name CRE_no_sgd --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=1 python train_CRE_mix.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_mix --name CRE_new—sgd --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=2 python train_CRE_vae.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_vae --name CRE_ttt --task CREMAD --task_type classification --model mml_av_vae --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=3 python train_CRE_mix_alter.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_D --name CRE_3 --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=3 python train_CRE_D.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_D --name CRE_softlabel --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=3 python train_CRE_D3.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_D3 --name CRE_4 --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=3 python train_CRE_latefusion.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_fusion --name CRE_latefusion --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100
