#!/bin/bash
set -e

# nyu - base
CUDA_VISIBLE_DEVICES=2 python train_nyu_IB.py --savedir ./saved/NYUD1 --name NYUD18 --max_epochs 100 --seed 1 --noise 0.0
