
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.bert import BertEncoder, BertClf
from backbone.resnet_AVT import resnet18



class MMLBase_vta_IB(nn.Module):
    def __init__(self, args):
        super(MMLBase_vta_IB, self).__init__()
        self.args = args
        self.audio = resnet18('audio')
        self.visual = resnet18('visual')
        self.txt_enc = BertClf(args)
        # self.txt_enc = BertEncoder(args)
        # self.audio_out = nn.Linear(768, args.n_classes)
        # self.visual_out = nn.Linear(768, args.n_classes)
        self.clf_head = self.txt_enc.clf(768, args.n_classes)


    def forward(self, txt, mask_t, segment_t, audio, visual):
        # aduio visual 的输出定义
        audio = audio.unsqueeze(1) #audio [B, 1, T,D]  visual [B, channel, fps, H, W]
        a = self.audio(audio)  
        v = self.visual(visual)
        t_hidden, t_logits = self.txt_enc(txt, mask_t, segment_t)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a_out = torch.flatten(a, 1)
        v_out = torch.flatten(v, 1)


        return a_out, v_out, t_logits, t_hidden
    



    






    