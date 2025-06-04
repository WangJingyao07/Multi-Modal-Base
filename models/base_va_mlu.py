import torch
import torch.nn as nn

from backbone.bert import BertEncoder, BertClf
from backbone.img import ImageEncoder, ImageClf

# from backbone.resnet_AV import resnet18


class MMLBase_va_mlu(nn.Module):
    def __init__(self, args):
        super(MMLBase_va_mlu, self).__init__()
        self.args = args
        self.txtclf = BertClf(args) 
        self.imgclf= ImageClf(args)

        self.mlu_head = nn.Linear(768, args.n_classes)

        self.clf_img = nn.Linear(args.img_hidden_sz * args.num_image_embeds, 768)  



    def forward(self, txt, mask, segment, img):
        
        
        txt_out = self.txtclf(txt, mask, segment)   
        img_out = self.imgclf(img)  
        img_out = self.clf_img(img_out)

        return txt_out, img_out






 
#
   

