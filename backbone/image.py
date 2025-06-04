import torch
import torch.nn as nn
import torchvision
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        """
        model = models.__dict__['resnet18'](num_classes=365)
        # places model downloaded from http://places2.csail.mit.edu/
        checkpoint = torch.load(args.CONTENT_MODEL_PATH, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        print('content model pretrained using place')
        """
        model = torchvision.models.resnet18(pretrained=False)
        state_dict = torch.load(args.CONTENT_MODEL_PATH)
        model.load_state_dict(state_dict)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        out = self.model(x)
        # print(out.shape)
        out = self.pool(out)
        # print(out.shape)
        out = torch.flatten(out, start_dim=2)
        # print(out.shape)
        out = out.transpose(1, 2).contiguous()
        return out  
