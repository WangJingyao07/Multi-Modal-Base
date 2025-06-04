#!/usr/bin/env python3
import argparse
import copy
import os
import random

import torch
import numpy as np
import torchaudio
import librosa
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer

from data.vocab import Vocab



def get_glove_words(path: str):
    """Load vocabulary words from a GloVe file."""
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            w, _ = line.split(" ", 1)
            words.append(w)
    return words


def wav2fbank(
    filename1: str,
    filename2: str = None,
    mix_lambda: float = -1
) -> torch.Tensor:
    """
    Load one or two waveforms, optionally mix them, then compute and
    pad/cut a Kaldi-style mel-filterbank.
    """
    # load and center
    wav1, sr = torchaudio.load(filename1)
    wav1 = wav1 - wav1.mean()
    if filename2:
        wav2, _ = torchaudio.load(filename2)
        wav2 = wav2 - wav2.mean()
        # align lengths
        L1, L2 = wav1.shape[1], wav2.shape[1]
        if L1 != L2:
            if L1 > L2:
                pad = torch.zeros(1, L1)
                pad[0, :L2] = wav2
                wav2 = pad
            else:
                wav2 = wav2[:, :L1]
        wav1 = mix_lambda * wav1 + (1 - mix_lambda) * wav2
        wav1 = wav1 - wav1.mean()

    try:
        fbank = torchaudio.compliance.kaldi.fbank(
            wav1,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10
        )
    except Exception:
        fbank = torch.full((512, 128), 0.01)
        print("Warning: failed to compute fbank, returning small constant")

    # pad or cut to 1024 frames
    T, _ = fbank.shape
    if T < 1024:
        pad = torch.nn.ZeroPad2d((0, 0, 0, 1024 - T))
        fbank = pad(fbank)
    else:
        fbank = fbank[:1024, :]

    return fbank



class AddGaussianNoise:
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean, self.variance, self.amplitude = mean, variance, amplitude

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        h, w, c = arr.shape
        noise = self.amplitude * np.random.normal(self.mean, self.variance, (h, w, 1))
        noise = np.repeat(noise, c, axis=2)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert('RGB')


class AddSaltPepperNoise:
    def __init__(self, density=0.05, p=0.5):
        self.density, self.p = density, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.array(img)
        h, w, c = arr.shape
        choices = [0, 255, None]
        probs = [self.density / 2, self.density / 2, 1 - self.density]
        mask = np.random.choice([0, 1, 2], size=(h, w), p=probs)
        out = arr.copy()
        out[mask == 0] = 0
        out[mask == 1] = 255
        return Image.fromarray(out).convert('RGB')




def get_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_noise_transforms(kind: str, severity: float) -> transforms.Compose:
    base = [transforms.Resize(256)]
    if kind.lower() == 'gaussian':
        base.append(transforms.RandomApply([AddGaussianNoise(amplitude=severity * 10)], p=0.5))
    elif kind.lower() == 'salt':
        base.append(transforms.RandomApply([AddSaltPepperNoise(density=severity/10, p=0.5)], p=0.5))
    base += [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(base)




class CREDataset(Dataset):
    """CREMAD audio¨Cvisual dataset."""

    def __init__(self, args, mode='train'):
        self.args, self.mode = args, mode
        root = args.data_path
        self.audio_dir = os.path.join(root, 'audio', mode)
        self.visual_dir = os.path.join(root, 'visual', f"{mode}_imgs/Image-01-FPS")
        self.labels = self._load_labels(args.stat_cre_path)
        self.file_list, self.label_map = self._scan_split(args, mode)
        self.transforms = get_transforms()

    def _load_labels(self, path):
        with open(path) as f:
            return [line.strip() for line in f]

    def _scan_split(self, args, mode):
        txt = args.train_txt if mode == 'train' else args.test_txt
        mapping = {}
        files = []
        for line in open(txt):
            base, lbl = (line.strip().split('.flv ') if args.dataset == 'CREMAD'
                         else line.strip().split('.mp4 '))
            wav = os.path.join(self.audio_dir, base + '.wav')
            img_dir = os.path.join(self.visual_dir, base)
            if os.path.isdir(img_dir) and os.path.isfile(wav):
                files.append(base)
                mapping[base] = lbl
        return files, mapping

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        key = self.file_list[idx]
        # audio
        spec = wav2fbank(os.path.join(self.audio_dir, f"{key}.wav"))
        # visual
        imgs = sorted(os.listdir(os.path.join(self.visual_dir, key)))
        picks = [imgs[i * len(imgs) // 3] for i in range(3)]
        tensors = []
        for p in picks:
            img = Image.open(os.path.join(self.visual_dir, key, p)).convert('RGB')
            t = self.transforms(img).unsqueeze(1)
            tensors.append(t)
        image_tensor = torch.cat(tensors, dim=1)
        label = self.labels.index(self.label_map[key])
        return spec, image_tensor, label, torch.LongTensor([idx])


class IEMODataset(Dataset):
    """IEMOCAP audio¨Cvisual¨Ctext dataset."""

    def __init__(self, args, tokenizer, vocab, transforms, mode='train'):
        self.args, self.mode = args, mode
        root = args.data_path
        self.audio_dir = os.path.join(root, 'audio', mode)
        self.visual_dir = os.path.join(root, 'visual', f"{mode}_imgs")
        self.tokenizer, self.vocab = tokenizer, vocab
        self.labels = sorted([l.strip() for l in open(args.stat_iemo_path)])
        self.samples = self._load_samples(mode)
        self.transforms = transforms
        self.start_token = "[CLS]" if args.model != "mmbt" else "[SEP]"

    def _load_samples(self, mode):
        path = {
            'train': self.args.train_iemo_txt,
            'dev':   self.args.dev_iemo_txt,
            'test':  self.args.test_iemo_txt
        }[mode]
        data = []
        for line in open(path):
            vid, txt, lbl = line.strip().split(" [split|sign] ")
            vid = vid.replace(".mp4", "")
            wav = os.path.join(self.audio_dir, vid + '.wav')
            img_dir = os.path.join(self.visual_dir, vid)
            if os.path.isdir(img_dir) and os.path.isfile(wav):
                data.append((vid, txt, lbl))
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, txt, lbl = self.samples[idx]
        # text
        tokens = [self.start_token] + self.tokenizer(txt)[:self.args.max_seq_len - 1]
        if self.args.noise > 0:
            tokens = self._apply_noise(tokens)
        seq = torch.LongTensor([self.vocab.stoi.get(t, self.vocab.stoi["[UNK]"]) for t in tokens])
        segment = torch.zeros(len(seq), dtype=torch.long)
        # visual
        imgs = sorted(os.listdir(os.path.join(self.visual_dir, vid)))
        img = Image.open(os.path.join(self.visual_dir, vid, imgs[len(imgs)//2])).convert('RGB')
        img = self.transforms(img)
        # audio
        y, sr = librosa.load(os.path.join(self.audio_dir, vid + '.wav'), sr=22050)
        y = np.tile(y, 3)[:sr * 3]
        y = np.clip(y, -1, 1)
        spec = torch.from_numpy(np.log(np.abs(librosa.stft(y, n_fft=512, hop_length=353)) + 1e-7))
        # label
        label = torch.LongTensor([self.labels.index(lbl)])
        return seq, segment, img, spec, label, torch.LongTensor([idx])

    def _apply_noise(self, tokens):
        out = []
        p_replace = self.args.noise / 10
        for w in tokens:
            if random.random() < p_replace:
                out.append('_')
            else:
                out.append(w)
        return out



def collate_fn(batch, args):
    texts, segs, imgs, specs, tgts, idxs = zip(*batch)
    max_txt = max(t.size(0) for t in texts)
    max_spec = max(s.size(1) for s in specs)
    bsz = len(batch)

    text_tensor = torch.zeros(bsz, max_txt, dtype=torch.long)
    segment_tensor = torch.zeros(bsz, max_txt, dtype=torch.long)
    mask_tensor = torch.zeros(bsz, max_txt, dtype=torch.long)
    spec_tensor = torch.zeros(bsz, specs[0].size(0), max_spec)
    img_tensor = torch.stack(imgs)
    tgt_tensor = torch.cat(tgts)
    idx_tensor = torch.cat(idxs).long()

    for i, (t, seg, spec) in enumerate(zip(texts, segs, specs)):
        L_t, L_s = t.size(0), spec.size(1)
        text_tensor[i, :L_t] = t
        segment_tensor[i, :L_t] = seg
        mask_tensor[i, :L_t] = 1
        spec_tensor[i, :, :L_s] = spec

    return text_tensor, segment_tensor, mask_tensor, img_tensor, spec_tensor, tgt_tensor, idx_tensor


def get_vocab(args):
    vocab = Vocab()
    if args.model in ["mml_vt","mml_avt", "latefusion", "mutual"]:
        btok = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        vocab.stoi, vocab.itos = btok.vocab, btok.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)
    else:
        vocab.add(get_glove_words(args.glove_path))
    return vocab


def get_data_loaders(args):
    # tokenizer choice
    tokenizer = (BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
                 if args.model.startswith("mml_") else str.split)
    base_tf = get_transforms()
    vocab = get_vocab(args)
    args.vocab, args.vocab_sz = vocab, vocab.vocab_sz

    # datasets
    train_set = IEMODataset(args, tokenizer, vocab, base_tf, mode='train')
    dev_set   = IEMODataset(args, tokenizer, vocab, base_tf, mode='dev')
    test_tf   = get_noise_transforms(args.noise_type, args.noise) if args.noise > 0 else base_tf
    test_set  = IEMODataset(args, tokenizer, vocab, test_tf, mode='test')

    # data loaders
    coll = functools.partial(collate_fn, args=args)
    dl_args = dict(batch_size=args.batch_size, num_workers=args.n_workers, collate_fn=coll)
    train_loader = DataLoader(train_set, shuffle=True, **dl_args)
    val_loader   = DataLoader(dev_set, shuffle=False, **dl_args)
    test_loader  = DataLoader(test_set, shuffle=False, **dl_args)

    return train_loader, val_loader, test_loader

def get_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--stat_cre_path", type=str, required=True)
    p.add_argument("--train_txt", type=str, required=True)
    p.add_argument("--test_txt", type=str, required=True)
    p.add_argument("--stat_iemo_path", type=str, required=True)
    p.add_argument("--train_iemo_txt", type=str, required=True)
    p.add_argument("--dev_iemo_txt", type=str, required=True)
    p.add_argument("--test_iemo_txt", type=str, required=True)
    p.add_argument("--bert_model", type=str, default="bert-base-uncased")
    p.add_argument("--glove_path", type=str, default="./glove.840B.300d.txt")
    p.add_argument("--model", type=str, default="mml_avt")
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--noise_type", type=str, default="Gaussian")
    p.add_argument("--n_workers", type=int, default=4)
    return p.parse_args()

