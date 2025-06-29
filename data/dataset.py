
import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.utils import truncate_seq_pair, numpy_seed

import random
class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        if args.model == "mmbt":
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.args.task == "vsnli":
            sent1 = self.tokenizer(self.data[index]["sentence1"])
            sent2 = self.tokenizer(self.data[index]["sentence2"])
            truncate_seq_pair(sent1, sent2, self.args.max_seq_len - 3)
            sentence = self.text_start_token + sent1 + ["[SEP]"] + sent2 + ["[SEP]"]
            segment = torch.cat(
                [torch.zeros(2 + len(sent1)), torch.ones(len(sent2) + 1)]
            )
        else:

            _ = self.tokenizer(self.data[index]["text"])
            if self.args.noise > 0.0:
                p = [0.5, 0.5]
                flag = np.random.choice([0, 1], p=p)
                if flag:
                    wordlist=self.data[index]["text"].split(' ')
                    for i in range(len(wordlist)):
                        replace_p=1/10*self.args.noise
                        replace_flag = np.random.choice([0, 1], p=[1-replace_p, replace_p])
                        if replace_flag:
                            wordlist[i]='_'
                    _=' '.join(wordlist)
                    _=self.tokenizer(_)
       

            sentence = (
                self.text_start_token
                + _[:(self.args.max_seq_len - 1)]
            )
            segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )
        # print(sentence)

        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
            ] = 1
        else:
            label = torch.LongTensor(
                [self.args.labels.index(self.data[index]["label"])]
            )

        image = None
        if self.args.model in ["mml_vt", "latefusion", "tmc"]:
            if self.data[index]["img"]:
                image = Image.open(
                    os.path.join(self.data_dir, self.data[index]["img"])
                ).convert("RGB")
            else:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)
        # if self.args.model == "mmbt":
        #     segment = segment[1:]
        #     sentence = sentence[1:]
        #     segment += 1
        # print(sentence.shape, segment.shape)
        # print(sentence.shape, segment.shape, image.shape, label.shape, torch.LongTensor([index]).shape)
        # print(type(sentence), type(segment), type(image), type(label), type(torch.LongTensor([index])))
        return sentence, segment, image, label, torch.LongTensor([index])

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):

        img = np.array(img)
        h, w, c = img.shape
        np.random.seed(0)
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                   
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class AddSaltPepperNoise(object):

    def __init__(self, density=0, p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  
            img = np.array(img) 
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd]) 
            mask = np.repeat(mask, c, axis=2) 
            img[mask == 0] = 0  
            img[mask == 1] = 255  
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img