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


class MVSADataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, tokenizer=None, max_length=128):
        self.root_dir = os.path.join(root_dir, 'mvsa')
        self.mode = mode
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = []
        with open(os.path.join(self.root_dir, f'{mode}.txt'), 'r') as f:
            for line in f:
                img_file, text, label = line.strip().split('\t')
                self.samples.append((img_file, text, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, text, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, 'images', img_file)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        token_type_ids = encoding['token_type_ids'].squeeze(0) if 'token_type_ids' in encoding else torch.zeros_like(input_ids)

        return input_ids, attention_mask, token_type_ids, image, label, torch.LongTensor([idx])

