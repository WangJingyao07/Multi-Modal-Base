#!/bin/bash
set -e

# food101 - base
python train_food.py --batch_sz 16 --gradient_accumulation_steps 40 --savedir ./saved/mml_vt_food101 --name mml_vt_food101 --task food101 --task_type classification --model mml_vt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1 --noise 0.0
