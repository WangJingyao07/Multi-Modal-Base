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

def wav2fbank(filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename,)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, 
                                                      use_energy=False, window_type='hanning', 
                                                      num_mel_bins=128, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')

        target_length = 1024
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank


class IEMODataset(Dataset):
    # audio and visual dataset
    def __init__(self, args, tokenizer, vocab, transforms, mode):
        classes = []
        data2class = {}

        self.mode = mode
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.data_root = args.data_path
        self.visual_path = os.path.join(self.data_root, "visual", '{}_imgs/'.format(mode))
        self.audio_path = os.path.join(self.data_root, "audio", '{}/'.format(mode))
        self.stat_path = "/data/users/wjy/data/IEMOCAP/stat_iemo.txt"
        self.train_txt = "/data/users/wjy/data/IEMOCAP/my_train_iemo.txt"
        self.test_txt = "/data/users/wjy/data/IEMOCAP/my_test_iemo.txt"
        self.eval_txt = "/data/users/wjy/data/IEMOCAP/my_dev_iemo.txt"
        self.max_seq_len = args.max_seq_len
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]
        self.transforms = transforms
        self.args = args

        with open(self.stat_path, "r") as f1:
            classes = f1.readlines()
        
        classes = [sclass.strip() for sclass in classes]

        if mode == 'train':
            csv_file = self.train_txt
        elif mode == 'test':
            csv_file = self.test_txt
        else:
            csv_file = self.eval_txt
        self.av_files = []
        self.content = []
        print(mode)

        with open(csv_file, "r") as f2:
            csv_reader = f2.readlines()
            for single_line in csv_reader:
                item = single_line.strip().split(" [split|sign] ")
                item[0] = item[0].split(".mp4")[0]
                txt_content = item[1]
                # print(txt_content)
                visual_path = os.path.join(self.visual_path, item[0])
                audio_path = os.path.join(self.audio_path, item[0] + '.wav')
                # print(visual_path)
                # print(audio_path)
          

                if txt_content and os.path.exists(visual_path) and os.path.exists(audio_path):
                    self.av_files.append(item[0])
                    self.content.append(txt_content)
                    data2class[item[0]] = item[-1]
                else:
                    continue

        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class

        # for item in data:
        #     self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

    def __len__(self):
        return len(self.av_files)
    
    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            if self.mode == "train":
                image_tensor = self.transforms(img)
            else:
                image_tensor = self.transforms(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.transforms(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.transforms(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def __getitem__(self, idx):

        av_file = self.av_files[idx]

        # Text
        _ = self.tokenizer(self.content[idx])
        
        # Ìí¼ÓÔëÉù
        if self.args.noise > 0.0:
            p = [0.5, 0.5]
            flag = np.random.choice([0, 1], p=p)
            if flag:
                wordlist=self.content[idx].split(' ')
                for i in range(len(wordlist)):
                    replace_p=1/10*self.args.noise
                    replace_flag = np.random.choice([0, 1], p=[1-replace_p, replace_p])
                    if replace_flag:
                        wordlist[i]='_'
                _=' '.join(wordlist)
                _=self.tokenizer(_)
       

        sentence = (self.text_start_token + _[:(self.max_seq_len - 1)])
        segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )
       
       

        # Visual
        visual_path = os.path.join(self.visual_path, av_file)
        # allimages = os.listdir(visual_path)
        # image = self.get_image(os.path.join(visual_path, allimages[int(len(allimages) / 2)]))


        image_samples = os.listdir(visual_path)
   
        select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.fps, 3, 224, 224))
        for i in range(self.args.fps):
            img = Image.open(os.path.join(visual_path, image_samples[i])).convert('RGB')
            img = self.transforms(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))
     
        # Audio
        audio_path = os.path.join(self.audio_path, av_file + '.wav')

        # fbank
        spectrogram = wav2fbank(audio_path, None, 0)


        label = torch.LongTensor([self.classes.index(self.data2class[av_file])])
      
        return sentence, segment, images, spectrogram, label, torch.LongTensor([idx])



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
