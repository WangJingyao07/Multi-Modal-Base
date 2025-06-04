

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.resnet_AV import resnet18
# from backbone.resnet_AVT import resnet18


class MMLBase_va(nn.Module):
    def __init__(self, args):
        super(MMLBase_va, self).__init__()
        self.args = args
        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        # self.audio_out = nn.Linear(512, args.n_classes)
        # self.visual_out = nn.Linear(512, args.n_classes)
    
        self.fusion_module = ConcatFusion(output_dim=args.n_classes)

        # self.mlu_head = nn.Linear(512, args.n_classes) #mlu算法


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
        # return a, v  #这个是mlu 要的输出 只要特征和一个mlu_head就行了
        




        
        # a_out = self.audio_out(a)
        # v_out = self.visual_out(v)
        # fusion = (a_out + v_out) / 2
        a, v, out = self.fusion_module(a, v)
        return out, a, v
    
        # a_v_out = (a_out + v_out) / 2
        # return a_v_out, a_out, v_out
        # # print("a_out", a_out.size())
        # # if self.args.mask ==  "soft_mask":
        # #     a_v_out = (a_out + v_out) * self.soft_mask / 2
        # #     return a_v_out, a_out, v_out, self.soft_mask    
        # # elif self.args.mask == "hard_mask":
        # #     a_v_out = (a_out + v_out) * self.hard_mask / 2
        # #     return a_v_out, a_out, v_out, self.hard_mask
        # # else:
        # #     a_v_out = (a_out + v_out) / 2
        # #     return a_v_out, a_out, v_out

        


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output





    