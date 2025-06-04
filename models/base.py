

import torch
import torch.nn as nn
from backbone.image import ImageEncoder
import torch.nn.functional as F



class MMLBase(nn.Module):
    def __init__(self, args):
        super(MMLBase, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder(args)
        self.depthenc = ImageEncoder(args)
        depth_last_size = args.img_hidden_sz * args.num_image_embeds
        rgb_last_size = args.img_hidden_sz * args.num_image_embeds
        self.clf_depth = nn.ModuleList()
        self.clf_rgb = nn.ModuleList()
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(depth_last_size, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_depth.append(nn.Linear(depth_last_size, args.n_classes))

        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(rgb_last_size, hidden))
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            rgb_last_size = hidden

        self.clf_rgb.append(nn.Linear(rgb_last_size, args.n_classes))

    
        # self.latent_mu = nn.Sequential(nn.Linear()) 


    def forward(self, rgb, depth):
        depth = self.depthenc(depth)
        depth = torch.flatten(depth, start_dim=1)
        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)
        
        rgb_out = rgb
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)
        
        depth_out = depth
        for layer in self.clf_depth:
            depth_out = layer(depth_out)


        depth_rgb_out = (depth_out + rgb_out) / 2
       

        return depth_rgb_out, rgb_out, depth_out
