#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python train_IEMO.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP --name iemo_1--task IEMOCAP --task_type classification --model mml_avt --fusion sum --patience 5 --lr 5e-05 --seed 1 --noise 0.0

CUDA_VISIBLE_DEVICES=2 python train_IEMO.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP --name iemo_fbank --task IEMOCAP --task_type classification --model mml_avt --patience 5 --lr 5e-05 --seed 1 --noise 0.0

CUDA_VISIBLE_DEVICES=2 python train_IEMO_MLU.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP_mlu --name iemo_no_sgd --task IEMOCAP --task_type classification --model mml_avt --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=3 python train_IEMO_mix.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP_mix --name iemo_no_sgd --task IEMOCAP --task_type classification --model mml_avt --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=1 python train_IEMO_D4.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP_D --name nofusionloss --task IEMOCAP --task_type classification --model mml_avt --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100 --Temp 1

CUDA_VISIBLE_DEVICES=1 python train_IEMO_MMRG.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP_MMRG --name iemo5 --task IEMOCAP --task_type classification --model mml_avt_mmrg --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=1 python train_IEMO_single.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP_single --name test --task IEMOCAP --task_type classification --model mml_avt --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100
