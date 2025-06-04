import pandas as pd
import cv2
import os
import pdb

class videoReader(object):
    def __init__(self, video_path, frame_interval=1, frame_kept_per_second=1):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.frame_kept_per_second = frame_kept_per_second

        self.vid = cv2.VideoCapture(self.video_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_len = int(self.video_frames/self.fps)


    def video2frame(self, frame_save_path):
        self.frame_save_path = frame_save_path
        success, image = self.vid.read()
        count = 0
        while success:
            count +=1
            if count % self.frame_interval == 0:
                save_name = '{}/frame_{}_{}.jpg'.format(self.frame_save_path, int(count/self.fps), count)  # filename_second_index
                cv2.imencode('.jpg', image)[1].tofile(save_name)
            success, image = self.vid.read()


    def video2frame_update(self, frame_save_path):
        self.frame_save_path = frame_save_path

        count = 0
        frame_interval = int(self.fps/self.frame_kept_per_second)
        while(count < self.video_frames):
            ret, image = self.vid.read()
            if not ret:
                break
            if count % self.fps == 0:
                frame_id = 0
            if frame_id<frame_interval*self.frame_kept_per_second and frame_id%frame_interval == 0:
                save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                cv2.imencode('.jpg', image)[1].tofile(save_name)

            frame_id += 1
            count += 1


class VGGSound_dataset(object):
    def __init__(self, path_to_dataset = '/data/users/wjy/data/Trian_datasets/CREMAD/visual', frame_interval=1, frame_kept_per_second=1, mode="train"):
        self.path_to_video = os.path.join(path_to_dataset, mode)
        self.frame_kept_per_second = frame_kept_per_second
        self.path_to_save = os.path.join(path_to_dataset, '{}_imgs'.format(mode), 'Image-{:02d}-FPS'.format(self.frame_kept_per_second))
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)

        videos = "/data/users/wjy/data/IEMOCAP/my_{}_iemo.txt".format(mode)
        # videos = 'my_{}_iemo.txt'.format(mode)
        with open(videos, 'r') as f:
            self.file_list = f.readlines()
        self.new_file_list = []

    def extractImage(self):

        for i, each_video in enumerate(self.file_list):
            if i % 100 == 0:
                print('*******************************************')
                print('Processing: {}/{}'.format(i, len(self.file_list)))
                print('*******************************************')
            video_name = each_video.strip().split()[0]
            video_dir = os.path.join(self.path_to_video, video_name)
            try:
                self.videoReader = videoReader(video_path=video_dir, frame_kept_per_second=self.frame_kept_per_second)

                # save_dir = os.path.join(self.path_to_save, video_name.split(".flv")[0])
                save_dir = os.path.join(self.path_to_save, video_name.split(".mp4")[0])

                # print(save_dir)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                self.videoReader.video2frame_update(frame_save_path=save_dir)
                self.new_file_list.append(each_video)
            except:
                print('Fail @ {}'.format(video_name))

        # with open("my_train_cre_new.txt", "w") as fl:
        #     fl.writelines(self.new_file_list)

if __name__ == "__main__":
    vggsound = VGGSound_dataset("/data/users/wjy/data/IEMOCAP/visual", mode="dev")
    vggsound.extractImage()