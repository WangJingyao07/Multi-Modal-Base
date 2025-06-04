import torch
import torch.nn as nn

from backbone.bert import BertEncoder, BertClf
from backbone.img import ImageEncoder, ImageClf
# from backbone.resnet_AV import resnet18



class MMLBase_vt(nn.Module):
    def __init__(self, args):
        super(MMLBase_vt, self).__init__()
        self.args = args
        self.txtclf = BertClf(args)
        self.imgclf= ImageClf(args)

        # self.img_head = nn.Linear(768, args.n_classes)

    def forward(self, txt, mask, segment, img):
        
        
        hidden_t, txt_logits = self.txtclf(txt, mask, segment)   
        hdden_i, img_logits = self.imgclf(img)  
        #    
        # print(img_out.shape)
        # img_logits = self.img_head(img_out) 
        # fusion_logits = self.mix_head((txt_out + img_out)/2)
        fusion_logits = (txt_logits + img_logits)/2
        return fusion_logits, txt_logits, img_logits
        # return txt_out, img_out
#
    # def forward(self, txt, mask, segment, img):
        
        
    #     txt_out = self.txtclf(txt, mask, segment)   
    #     img_out = self.imgclf(img)      

    #     mu_a = self.mu(txt_out)
    #     logvar_a = self.logvar(txt_out)

    #     eps = torch.randn_like(logvar_a)
        
    #     latent_a = mu_a + eps * torch.exp(logvar_a * 0.5)


    #     mu_v = self.mu(img_out)
    #     logvar_v = self.logvar(img_out)

    #     latent_v = mu_v + eps * torch.exp(logvar_v * 0.5)   
        
    #     logits_a = self.classifier(latent_a)
    #     logits_v = self.classifier(latent_v)

    #     return logits_a, mu_a, logvar_a, logits_v, mu_v, logvar_v
    #     # return logits, mu_txt, logvar_txt, mu_img, logvar_img

