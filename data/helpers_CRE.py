
import functools
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.dataset_AFFECTION import CREDataset, AddGaussianNoise, AddSaltPepperNoise


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

def get_GaussianNoisetransforms(rgb_severity):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomApply([AddGaussianNoise(amplitude=rgb_severity * 10)], p=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            ),
        ]
    )

def get_SaltNoisetransforms(rgb_severity):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomApply([AddSaltPepperNoise(density=0.1, p=rgb_severity/10)], p=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            ),
        ]
    )



def get_data_loaders(args):
   
    transforms = get_transforms()


    train = CREDataset(args, transforms, mode='train') 

    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers, pin_memory=True)


    if args.noise>0.0:
        if args.noise_type=='Gaussian':
            print('Gaussian')
            test_transforms=get_GaussianNoisetransforms(args.noise)
        elif args.noise_type=='Salt':
            print("Salt")
            test_transforms = get_SaltNoisetransforms(args.noise)
    else:
        test_transforms=transforms

        test = CREDataset(args, test_transforms, mode='test')

        test_loader = DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers, pin_memory=True)


    return train_loader, test_loader



import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/data/users/wjy/data/Affections/CREMAD')

    parser.add_argument("--bert_model", type=str, default="./bert-base-uncased")#
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--mask_percent', default=0.6, type=float, help='mask percent in training')
    parser.add_argument('--LOAD_SIZE', default=256, type=int)
    parser.add_argument('--FiNE_SIZE', default=224, type=int)
    parser.add_argument('--model', default='mml_avt', type=str, choices=['mml_vt', 'mml_avt', 'concatbert', 'mmbt', 'latefusion', 'tmc'])
    parser.add_argument("--glove_path", type=str, default="./datasets/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--noise_type", type=str, default="Gaussian")
    parser.add_argument("--n_workers", type=int, default=4)

    return parser.parse_args()

from tqdm import tqdm


if __name__ == "__main__":

    args = get_arguments()
    train_loader, test_loader = get_data_loaders(args)
    for batch in tqdm(train_loader, total=len(train_loader)):
        spec, img, label, i = batch
        print(spec.size(), img.size(), label.size())
