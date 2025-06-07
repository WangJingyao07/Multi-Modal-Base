import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.bert import BertEncoder, BertClf
from backbone.img import ImageEncoder, ImageClf

class MMLBase_vt(nn.Module):

    def __init__(self, args):
        super(MMLBase_vt, self).__init__()
        self.args = args
        
        self.txtclf = BertClf(args)
        self.imgclf = ImageClf(args)
        
        d = args.hidden_size
        
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)
        
        self.calibrator = nn.Linear(2 * d, d)
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
        self.c3_metrics = {}

    def forward(self, txt, mask, segment, img, labels=None):
        hidden_t, txt_logits = self.txtclf(txt, mask, segment)
        hidden_i, img_logits = self.imgclf(img)
        fusion_logits = (txt_logits + img_logits) / 2
        
        if labels is not None:
            B, d = hidden_t.size()
            Q = self.W_Q(hidden_t)
            K = self.W_K(hidden_i)
            scores = torch.sum(Q * K, dim=-1) / (d ** 0.5)
            weights = torch.sigmoid(scores).unsqueeze(-1)
            V = weights * self.W_V(hidden_i) + (1 - weights) * self.W_V(hidden_t)
            
            Z_cat = torch.cat([hidden_t, hidden_i], dim=-1)
            Z_c = self.calibrator(Z_cat)
            
            Z_c_cf = Z_c.clone().detach().requires_grad_(True)
            logits_cf = self.imgclf.classifier(Z_c_cf)
            loss_ce = self.ce_loss(logits_cf, labels)
            grad = torch.autograd.grad(loss_ce, Z_c_cf, retain_graph=True)[0]
            Z_bar = Z_c_cf - grad
            
            p_z = F.log_softmax(Z_cat, dim=-1)
            p_v = F.softmax(V, dim=-1)
            # p_v = F.softmax(V, dim=-1)


            loss_v = self.kl_loss(p_z, p_v)
            loss_fe = self.mse_loss(Z_bar, Z_c_cf)
            
            logits_bar = self.imgclf.classifier(Z_bar)
            preds_cf = logits_cf.argmax(dim=1)
            preds_bar = logits_bar.argmax(dim=1)
            risk_suff = (preds_cf != labels).float().mean()
            risk_nec = (preds_bar == labels).float().mean()
            loss_c3 = risk_suff + risk_nec
            
            total_loss = (
                loss_ce
                + self.args.lambda_v * loss_v
                + self.args.lambda_fe * loss_fe
                + loss_c3
            )
            self.c3_metrics = {
                'loss_ce': loss_ce,
                'loss_v': loss_v,
                'loss_fe': loss_fe,
                'loss_c3': loss_c3,
                'total_loss': total_loss
            }
        
        return fusion_logits, txt_logits, img_logits
