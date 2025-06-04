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


class BraTSDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = os.path.join(root_dir, 'brats')
        self.mode = mode
        self.transform = transform
        self.img_paths = glob(os.path.join(self.root_dir, mode, '*_t1ce.npy'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        seg_path = img_path.replace('_t1ce.npy', '_seg.npy')
        img = np.load(img_path).astype(np.float32)
        seg = np.load(seg_path).astype(np.int64)

        img = torch.tensor(img).unsqueeze(0)  # Add channel dim
        seg = torch.tensor(seg)

        if self.transform:
            img = self.transform(img)

        return img, seg, torch.LongTensor([idx])
