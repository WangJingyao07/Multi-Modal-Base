
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.bert import BertEncoder, BertClf
from backbone.resnet_AVT import resnet18



class MMRG(nn.Module):
    def __init__(self, args):
        super(MMRG, self).__init__()
        self.args = args
        self.audio = resnet18('audio')
        self.visual = resnet18('visual')
        self.txt_enc = BertClf(args)
        # self.txt_enc = BertEncoder(args)
        self.audio_out = nn.Linear(768, args.n_classes)
        self.visual_out = nn.Linear(768, args.n_classes)
        self.clf_txt = nn.Linear(768, args.n_classes)
        self.cos_fix = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.cos_learn = Simillarity(args, 768)

        self.meta_features = {mod: None for mod in ['text', 'audio', 'image']}


    def forward(self, txt, mask_t, segment_t, audio, visual):
        audio = audio.unsqueeze(1) #audio [B, 1, T,D]  visual [B, channel, fps, H, W]
        a = self.audio(audio)  
        v = self.visual(visual)
        t_out = self.txt_enc(txt, mask_t, segment_t)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a_out = torch.flatten(a, 1)
        v_out = torch.flatten(v, 1)
        
        meta_audio = torch.mean(a_out, dim=0)
        meta_visual = torch.mean(v_out, dim=0)
        meta_txt = torch.mean(t_out, dim=0)

        if self.training:
            if self.meta_features['text'] is None or self.meta_features['audio'] is None or self.meta_features['image'] is None:
                self.meta_features['text'] = meta_txt.detach()
                self.meta_features['audio'] = meta_audio.detach()
                self.meta_features['image'] = meta_visual.detach()
            else:
                self.meta_features['text'] = (self.meta_features['text'] + meta_txt.detach()) /2
                self.meta_features['audio'] = (self.meta_features['audio'] + meta_audio.detach()) /2
                self.meta_features['image'] = (self.meta_features['image'] + meta_visual.detach()) /2
            meta_txt = self.meta_features['text']
            meta_audio = self.meta_features['audio']        
            meta_visual = self.meta_features['image']    
        else:
            meta_txt = self.meta_features['text']
            meta_audio = self.meta_features['audio']
            meta_visual = self.meta_features['image']

        meta_txt = meta_txt.tile(B, 1)
        meta_audio = meta_audio.tile(B, 1)
        meta_visual = meta_visual.tile(B, 1)

        ta_fix = self.cos_fix(t_out, meta_audio).unsqueeze(1)
        ta_learn = self.cos_learn(t_out, meta_audio)
        tv_fix = self.cos_fix(t_out, meta_visual).unsqueeze(1)
        tv_learn = self.cos_learn(t_out, meta_visual)

        ta_related = (ta_fix + ta_learn) / 2
        tv_related = (tv_fix + tv_learn) / 2



        at_fix = self.cos_fix(a_out, meta_txt).unsqueeze(1)
        at_learn = self.cos_learn(a_out, meta_txt) 
        av_fix = self.cos_fix(a_out, meta_visual).unsqueeze(1)
        av_learn = self.cos_learn(a_out, meta_visual)
        at_related = (at_fix + at_learn) / 2
        av_related = (av_fix + av_learn) / 2

       
        vt_fix = self.cos_fix(v_out, meta_txt).unsqueeze(1)
        vt_learn = self.cos_learn(v_out, meta_txt)
        va_fix = self.cos_fix(v_out, meta_audio).unsqueeze(1)
        va_learn = self.cos_learn(v_out, meta_audio)

        vt_related = (vt_fix + vt_learn) / 2
        va_related = (va_fix + va_learn) / 2

        # return a_out, v_out, t_out
    

        a_logits = self.audio_out(a_out)
        v_logits = self.visual_out(v_out)
        t_logits = self.clf_txt(t_out)

        adap_a = at_related * self.clf_txt(a_out) + av_related * self.visual_out(a_out)
        adap_v = vt_related * self.clf_txt(v_out) + va_related * self.audio_out(v_out)
        adap_t = ta_related * self.audio_out(t_out) + tv_related * self.visual_out(t_out)

        fusion_out = (a_logits + v_logits + t_logits + adap_a + adap_v + adap_t) / 6
        

        return fusion_out, a_logits, t_logits, v_logits, adap_a, adap_t, adap_v
    


class Simillarity(nn.Module):
    def __init__(self, args, idim=768):
        super(Simillarity, self).__init__()
        self.args = args
  
   
        self.weight_vectors = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(768, 768)) for _ in range(4)
        ])
        
        self.cos_learn = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weight_vectors:
            torch.nn.init.xavier_uniform_(weight)

    def forward(self, node_features1, node_features2):
        
        similarities = []
    
        
        for weight in self.weight_vectors:
            # Compute weighted node features for both domains
            weighted_features1 = torch.matmul(node_features1, weight)
            weighted_features2 = torch.matmul(node_features2, weight)
            # Compute cosine similarity between nodes from different domains
            similarity_matrix = self.cos_learn(weighted_features1, weighted_features2)
            # print(similarity_matrix.shape)
            similarities.append(similarity_matrix)
        # Average over all heads
        final_similarity = sum(similarities) / 4
        return final_similarity.unsqueeze(1)



    