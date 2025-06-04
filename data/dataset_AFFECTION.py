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

class CREDataset(Dataset):
    # audio and visual dataset
    def __init__(self, args, transforms, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        self.args = args
        self.data_root = args.data_path
        self.visual_path = os.path.join(self.data_root, "visual/", '{}_imgs/Image-01-FPS/'.format(mode))
        self.audio_path = os.path.join(self.data_root, "audio/", '{}/'.format(mode))
        self.stat_path = "/data/users/wjy/data/CREMAD/stat_cre.txt"
        self.train_txt = "/data/users/wjy/data/CREMAD/my_train_cre.txt"
        self.test_txt = "/data/users/wjy/data/CREMAD/my_test_cre.txt"
        self.transforms = transforms


   
        with open(self.stat_path, "r") as f1:
            classes = f1.readlines()
            classes = [sclass.strip() for sclass in classes]
        

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(csv_file, "r") as f2:
            # csv_reader = csv.reader(f2)
            csv_reader = f2.readlines()
            for single_line in csv_reader:
                # if args.dataset == "CREMAD":
                #     item = single_line.strip().split(".flv ")
                # else:
                #     item = single_line.strip().split(".mp4 ")
                item = single_line.strip().split(".flv ")

                audio_path = os.path.join(self.audio_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_path, item[0])
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    data.append(item[0])
                    data2class[item[0]] = item[1]
                else:
                    continue
        # print(data)

        self.classes = sorted(classes)

        # print(self.classes)
        self.data2class = data2class
        self.av_files = []
        for item in data:
            self.av_files.append(item)
            
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

    def __len__(self):
        return len(self.av_files)

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Audio
        audio_path = os.path.join(self.audio_path, av_file + '.wav')
        samples, rate = librosa.load(audio_path, sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)  # [257 188]


        # Visual
        visual_path = os.path.join(self.visual_path, av_file)
        allimages = os.listdir(visual_path)
        file_num = len(allimages)

        pick_num = 3
        seg = int(file_num / pick_num)
        image_arr = []

        for i in range(pick_num):
            tmp_index = int(seg * i)
            image = Image.open(os.path.join(visual_path, allimages[tmp_index])).convert('RGB')
            image = self.transforms(image)
            image = image.unsqueeze(1).float()
            image_arr.append(image)
            if i == 0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)


        label = self.classes.index(self.data2class[av_file])
        return spectrogram, image_n, label, torch.LongTensor([idx])

