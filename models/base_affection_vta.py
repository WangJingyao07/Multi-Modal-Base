

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.bert import BertEncoder, BertClf
from backbone.resnet_AV import resnet18



class MMLBase_affection_vta(nn.Module):
    def __init__(self, args):
        super(MMLBase_affection_vta, self).__init__()
        self.args = args
        self.audio = resnet18('audio')
        self.visual = resnet18('visual')
        self.txt_enc = BertClf(args)
        self.audio_out = nn.Linear(512, args.n_classes)
        self.visual_out = nn.Linear(512, args.n_classes)



    def forward(self, txt, mask_t, segment_t, audio, visual):
        # aduio visual 的输出定义
        audio = audio.unsqueeze(1) #audio [B, 1, T,D]  visual [B, channel, fps, H, W]
        a = self.audio_net(audio)  
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a_out = torch.flatten(a, 1)
        v_out = torch.flatten(v, 1)

        a_out = self.audio_out(a_out)
        v_out = self.visual_out(v_out)
        t_out = self.txt_enc(txt, mask_t, segment_t)


        fusion_out = (a_out + v_out + t_out) / 3
        

        return fusion_out, a_out, v_out, t_out
    


    