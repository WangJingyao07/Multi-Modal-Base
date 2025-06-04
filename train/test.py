import argparse
import torch
import numpy as np

import torchaudio


def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=16)
    parser.add_argument("--bert_model", type=str, default="./bert-base-uncased")#
    parser.add_argument("--data_path", type=str, default="/data/users/wjy/mmlbase/QMF/datasets/nyud2_trainvaltest")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--hidden_sz", type=int, default=768)

    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=40)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--name", type=str, default="my_model3")
    parser.add_argument("--model", type=str, default="mml_base")
    parser.add_argument("--num_image_embeds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--savedir", type=str, default="./saved/NYUD3/5")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--CONTENT_MODEL_PATH", type=str, default="./checkpoint/resnet18_pretrained.pth")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--noise_type", type=str, default="Gaussian")

import torch.nn.functional as F
import json
if __name__ == '__main__':

    t = torch.tensor([1,2,3,4,5])
