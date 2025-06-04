

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.bert import BertEncoder, BertClf
from backbone.resnet_AV import resnet18




class MMLBase_affection_va(nn.Module):
    def _init__(self, args):
        
        super(MMLBase_affection_va, self).__init__()
        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')
        self.audio_out = nn.Linear(512, args.n_classes)
        self.visual_out = nn.Linear(512, args.n_classes)

    def forward(self, audio, visual):
        
        audio = audio.unsqueeze(1) #[B, 1, T,D]  visual [B, channel, fps, H, W]
        
        a = self.audio_net(audio)  
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        a_out = self.audio_out(a)
        v_out = self.visual_out(v)
        a_v_out = (a_out + v_out) / 2

        return a_v_out, a_out, v_out




    