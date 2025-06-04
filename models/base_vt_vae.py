
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.resnet_AV import resnet18
# from backbone.resnet_AVT import resnet18


class MMLBase_av_vae(nn.Module):
    def __init__(self, args):
        super(MMLBase_av_vae, self).__init__()
        self.args = args
        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        self.mu = nn.Linear(512, 512)
        self.logvar = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.softplux = nn.Softplus()


        self.classifier = nn.Linear(512, args.n_classes)

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

        mu_a = self.mu(self.relu(a))
        logvar_a = self.softplux(self.logvar(a))

        eps = torch.randn_like(logvar_a)
        
        latent_a = mu_a + eps * torch.exp(logvar_a * 0.5)


        mu_v = self.mu(self.relu(v))
        logvar_v =  self.softplux(self.logvar(v))

        latent_v = mu_v + eps * torch.exp(logvar_v * 0.5)   
        
        logits_a = self.classifier(latent_a)
        logits_v = self.classifier(latent_v)

        return logits_a, mu_a, logvar_a, logits_v, mu_v, logvar_v




    
        







    