import os
from random import shuffle
data_dir = "/data/users/wjy/data/MLAUA/IEMOCAP"

as_info = []
train_txt, dev_txt, test_txt = [], [], []


for i in range(1,6):
    wav_path = os.path.join(data_dir, "Session{}".format(i), "sentences", "wav")
    for root, dirs, files in os.walk(wav_path):
        if root.endswith("wav"):
            continue
            # print("root",root)
        else:
            for  file in files:
                wav_file = os.path.join(root, file)
                as_info.append(wav_file)

shuffle(as_info)

all_len = len(as_info)
train_len = int(0.6 * all_len)
dev_len = train_len + int(0.2*all_len)

train_info = as_info[:train_len]
dev_info = as_info[train_len:dev_len]
test_info = as_info[dev_len:]

for line in train_info:
    path, caption, label = line.split(" [split|sign] ")
    filename = path.split("/")[-1]
    train_txt.append("{} [split|sign] {} [split|sign] {}".format(filename.replace(".wav", ".mp4"), caption, label))
for line in dev_info:
    path, caption, label = line.split(" [split|sign] ")
    filename = path.split("/")[-1]
    dev_txt.append("{} [split|sign] {} [split|sign] {}".format(filename.replace(".wav", ".mp4"), caption, label))
for line in test_info:
    path, caption, label = line.split(" [split|sign] ")
    filename = path.split("/")[-1]
    test_txt.append("{} [split|sign] {} [split|sign] {}".format(filename.replace(".wav", ".mp4"), caption, label))

with open("my_train_iemo.txt", "w") as mf:
    mf.writelines(train_txt)
with open("my_dev_iemo.txt", "w") as mf:
    mf.writelines(dev_txt)
with open("my_test_iemo.txt", "w") as mf:
    mf.writelines(test_txt)