
import argparse
import functools
import json
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer

from data.dataset_AFFECTION import IEMODataset, AddGaussianNoise, AddSaltPepperNoise
from data.vocab import Vocab


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/data/users/wjy/data/Affections/IEMOCAP')
    parser.add_argument("--bert_model", type=str, default="./bert-base-uncased")
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--mask_percent', default=0.6, type=float)
    parser.add_argument('--LOAD_SIZE', default=256, type=int)
    parser.add_argument('--FiNE_SIZE', default=224, type=int)
    parser.add_argument('--model', default='mml_avt', type=str, choices=['mml_vt', 'mml_avt', 'concatbert', 'mmbt', 'latefusion', 'tmc'])
    parser.add_argument("--glove_path", type=str, default="./datasets/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--noise_type", type=str, default="Gaussian")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--fps", type=int, default=4)
    return parser.parse_args()


def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_GaussianNoisetransforms(severity):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomApply([AddGaussianNoise(amplitude=severity * 10)], p=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_SaltNoisetransforms(severity):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomApply([AddSaltPepperNoise(density=0.1, p=severity / 10)], p=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    if args.model in ["IB_AVT", "IB_AVT_MLU", "latefusion", "mutual"]:
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)
    else:
        word_list = get_glove_words(args.glove_path)
        vocab.add(word_list)
    return vocab


def collate_fn(batch, args):
    lens = [len(row[0]) for row in batch]
    lens_spec = [row[3].size()[1] for row in batch]
    spec_dim = batch[0][3].shape[0]
    max_seq_len = max(lens)
    max_spec_len = max(lens_spec)
    bsz = len(batch)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()
    spec_tensor = torch.zeros([bsz, spec_dim, max_spec_len])
    img_tensor = torch.stack([row[2] for row in batch])
    tgt_tensor = torch.cat([row[4] for row in batch])
    idx_tensor = torch.cat([row[5] for row in batch]).long()

    for i, (row, l_txt, l_spec) in enumerate(zip(batch, lens, lens_spec)):
        text_tensor[i, :l_txt] = row[0]
        segment_tensor[i, :l_txt] = row[1]
        mask_tensor[i, :l_txt] = 1
        spec_tensor[i, :, :l_spec] = row[3]

    return text_tensor, segment_tensor, mask_tensor, img_tensor, spec_tensor, tgt_tensor, idx_tensor


def get_data_loaders(args):
    tokenizer = (
        BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        if args.model in ["IB_AVT", "IB_AVT_MLU", "latefusion", "tmc"]
        else str.split
    )
    base_transforms = get_transforms()
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz

    train = IEMODataset(args, tokenizer, vocab, base_transforms, mode='train')
    dev = IEMODataset(args, tokenizer, vocab, base_transforms, mode='dev')

    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, collate_fn=collate)
    val_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, collate_fn=collate)

    if args.noise > 0.0:
        if args.noise_type == 'Gaussian':
            print('Applying Gaussian noise')
            test_transforms = get_GaussianNoisetransforms(args.noise)
        elif args.noise_type == 'Salt':
            print("Applying Salt & Pepper noise")
            test_transforms = get_SaltNoisetransforms(args.noise)
    else:
        test_transforms = base_transforms

    test = IEMODataset(args, tokenizer, vocab, test_transforms, mode='test')
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, collate_fn=collate)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    args = get_arguments()
    train_loader, val_loader, test_loader = get_data_loaders(args)

    for i, (text, segment, mask, img, spec, tgt, idx) in enumerate(train_loader):
        print("Batch shapes:")
        print("Text:", text.shape, "Segment:", segment.shape, "Mask:", mask.shape)
        print("Image:", img.shape, "Spectrogram:", spec.shape, "Label:", tgt.shape, "Index:", idx.shape)
        break
