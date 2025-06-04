import os
import pickle

from random import shuffle
data_dir = "/data/users/wjy/data/MLAUA/IEMOCAP"

as_info = []
train_txt, dev_txt, test_txt = [], [], []
neu, hap, ang, sad = [], [], [], []
fru, exc = [], []


label_map = {0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry'}

wavs = []
label_pkl = '/data/users/wjy/data/MLAUA/IEMOCAP/IEMOCAP_features_raw_4way.pkl'
nums =  0
videoIDs, videoLabels, videoSpeakers, videoSentences, trainVid, testVid = pickle.load(open(label_pkl, "rb"), encoding='latin1')



for dicts in videoLabels:
    combined_list = [a + ".mp4 [split|sign] " + b + " [split|sign] " + label_map[c] for a, b, c in zip(videoIDs[dicts], videoSentences[dicts], videoLabels[dicts])]
    # as_info.append(combined_list) 

    as_info += combined_list

all_len = len(as_info)
train_len = int(0.6 * all_len)
dev_len = train_len + int(0.2*all_len)

shuffle(as_info)
train_info = as_info[:train_len]
dev_info = as_info[train_len:dev_len]
test_info = as_info[dev_len:]

with open("my_train_iemo.txt", "w") as mf:
    for item in train_info:
        mf.write("%s\n" % item)
    mf.writelines(train_txt)
with open("my_dev_iemo.txt", "w") as mf:
    for item in dev_info:
        mf.write("%s\n" % item) 
    # mf.writelines(dev_txt)
with open("my_test_iemo.txt", "w") as mf:
    # mf.writelines(test_txt)
    for item in test_info:
        mf.write("%s\n" % item) 
