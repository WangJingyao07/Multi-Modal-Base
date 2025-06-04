#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0 python train_food.py --batch_sz 16 --gradient_accumulation_steps 40 --savedir ./saved/food101 --name IB_VT02 --task food101  --task_type classification --model mml_vt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1 --noise 0.0

CUDA_VISIBLE_DEVICES=1 python train_mvsa.py --batch_sz 16 --gradient_accumulation_steps 40 --savedir ./saved/MVSA_VAE --name MVSA_Single_vaenokl --task MVSA_Single --task_type classification --model mml_vt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1 --noise 0.0 --dataset MVSA

CUDA_VISIBLE_DEVICES=2 python train_mvsa_D.py --batch_sz 16 --gradient_accumulation_steps 40 --savedir ./saved/MVSA_D --name MVSA_New2 --task MVSA_Single --task_type classification --model mml_vt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1 --noise 0.0 --dataset MVSA

CUDA_VISIBLE_DEVICES=3 python train_mvsa_D3.py --batch_sz 16 --gradient_accumulation_steps 40 --savedir ./saved/MVSA_D --name MVSA_New2 --task MVSA_Single --task_type classification --model mml_vt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1 --noise 0.0 --dataset MVSA

CUDA_VISIBLE_DEVICES=4 python train_mvsa_fusion.py --batch_sz 16 --gradient_accumulation_steps 40 --savedir ./saved/MVSA_fusion --name MVSA_Single6 --task MVSA_Single --task_type classification --model mml_vt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1 --noise 0.0 --dataset MVSA

CUDA_VISIBLE_DEVICES=5 python train_mvsa_D3.py --batch_sz 16 --gradient_accumulation_steps 40 --savedir ./saved/MVSA_D3 --name MVSA_4 --task MVSA_Single --task_type classification --model mml_vt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1 --noise 0.0 --dataset MVSA

CUDA_VISIBLE_DEVICES=6 python train_mvsamlu.py --batch_sz 16 --gradient_accumulation_steps 40 --savedir ./saved/MVSA_MLU --name MVSA_1 --task MVSA_Single --task_type classification --model mml_avt_mlu --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1 --noise 0.0 --dataset MVSA

CUDA_VISIBLE_DEVICES=7 python train_nyu_IB.py --savedir ./saved/NYUD1 --name NYUD18 --max_epochs 100 --seed 1 --noise 0.0

CUDA_VISIBLE_DEVICES=0 python train_CRE.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE --name CRE_librosa --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0

CUDA_VISIBLE_DEVICES=1 python train_CRE_OGM.py --batch_size 64 --gradient_accumulation_steps 40 --savedir ./saved/CRE_oge --name CRE_10 --task CREMAD --task_type classification --model mml_av --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=2 python train_CRE_MLU.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_MLU --name CRE_no_sgd --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=3 python train_CRE_mix.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_mix --name CRE_new—sgd --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=4 python train_CRE_vae.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_vae --name CRE_ttt --task CREMAD --task_type classification --model mml_av_vae --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=5 python train_CRE_mix_alter.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_D --name CRE_3 --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=6 python train_CRE_D.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_D --name CRE_softlabel --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=7 python train_CRE_D3.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_D3 --name CRE_4 --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=0 python train_CRE_latefusion.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/CRE_fusion --name CRE_latefusion --task CREMAD --task_type classification --model mml_av --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=1 python train_IEMO.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP --name iemo_1 --task IEMOCAP --task_type classification --model mml_avt --fusion sum --patience 5 --lr 5e-05 --seed 1 --noise 0.0

CUDA_VISIBLE_DEVICES=2 python train_IEMO.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP --name iemo_fbank --task IEMOCAP --task_type classification --model mml_avt --patience 5 --lr 5e-05 --seed 1 --noise 0.0

CUDA_VISIBLE_DEVICES=3 python train_IEMO_MLU.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP_mlu --name iemo_no_sgd --task IEMOCAP --task_type classification --model mml_avt --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=4 python train_IEMO_mix.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP_mix --name iemo_no_sgd --task IEMOCAP --task_type classification --model mml_avt --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=5 python train_IEMO_D4.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP_D --name nofusionloss --task IEMOCAP --task_type classification --model mml_avt --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100 --Temp 1

CUDA_VISIBLE_DEVICES=6 python train_IEMO_MMRG.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP_MMRG --name iemo5 --task IEMOCAP --task_type classification --model mml_avt_mmrg --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100

CUDA_VISIBLE_DEVICES=7 python train_IEMO_single.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP_single --name test --task IEMOCAP --task_type classification --model mml_avt --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100
