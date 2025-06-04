

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.bert import BertEncoder, BertClf
from backbone.resnet_AVT import resnet18



class MMLBase_VTA(nn.Module):
    def __init__(self, args):
        super(MMLBase_VTA, self).__init__()
        self.args = args
        self.audio = resnet18('audio')
        self.visual = resnet18('visual')
        self.txt_enc = BertClf(args)
        # self.txt_enc = BertEncoder(args)
        self.audio_out = nn.Linear(768, args.n_classes)
        self.visual_out = nn.Linear(768, args.n_classes)
        # self.clf_txt = self.txt_enc.clf(768, args.n_classes)
        self.private = nn.Linear(768, 768)
        self.Temp = args.Temp


    def forward(self, txt, mask_t, segment_t, audio, visual):
        hints_loss = {}
        audio = audio.unsqueeze(1) #audio [B, 1, T,D]  visual [B, channel, fps, H, W]
        a = self.audio(audio)  
        v = self.visual(visual)
        hidden, t_out = self.txt_enc(txt, mask_t, segment_t)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a_out = torch.flatten(a, 1)
        v_out = torch.flatten(v, 1)

        txt_loss = self.embedding_loss(a_out, v_out, hidden)
        audio_loss = self.embedding_loss(hidden,v_out, a_out)
        visual_loss = self.embedding_loss(hidden,a_out, v_out)
        hints_loss['txt_teacher'] = txt_loss
        hints_loss['audio_teacher'] = audio_loss    
        hints_loss['visual_teacher'] = visual_loss
        

        a_out = self.audio_out(a_out)
        v_out = self.visual_out(v_out)
        # t_out = self.clf_txt(t_out)




        fusion = (a_out + v_out + t_out) / 3
        

        return fusion, a_out, v_out, t_out, hints_loss
    
    def embedding_loss(self, s1_hidden, s2_hidden, t_hidden):
        """
        Compute the embedding loss between two hidden states.
        """
        # Normalize the hidden states
        s_pro1 = self.private(s1_hidden)
        s_pro2 = self.private(s2_hidden)
        s_pro1 = nn.functional.log_softmax(s_pro1/self.Temp, dim=1)
        s_pro2 = nn.functional.log_softmax(s_pro2/self.Temp, dim=1)
        t_hidden = nn.functional.softmax(t_hidden/self.Temp, dim=1)
        # Compute the KL divergence
        loss_s1 = F.kl_div(s_pro1, t_hidden, reduction='batchmean')
        loss_s2 = F.kl_div(s_pro2, t_hidden, reduction='batchmean')

        # loss = F.mse_loss(s_pro, t_hidden)

        return loss_s1 + loss_s2
    






    