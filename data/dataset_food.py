import copy
import os
import torch
import torchaudio
from PIL import Image
import random
import argparse
import numpy as np
import librosa
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob


class Food101Dataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.mode = mode
        self.root_dir = os.path.join(root_dir, 'food101')
        self.img_dir = os.path.join(self.root_dir, 'images')
        self.label_file = os.path.join(self.root_dir, f'{mode}.txt')
        self.transform = transform

        with open(self.label_file, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]

        self.classes = sorted(set([x.split('/')[0] for x in self.samples]))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        img_path = os.path.join(self.img_dir, sample_path + '.jpg')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[sample_path.split('/')[0]]
        return image, label, torch.LongTensor([idx])
