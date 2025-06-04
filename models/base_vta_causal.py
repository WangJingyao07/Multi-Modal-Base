
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.bert import BertClf
from backbone.resnet_AVT import resnet18


class MMLBase_vta_late(nn.Module):
    def __init__(self, args):
        super(MMLBase_vta_late, self).__init__()
        self.args = args

        self.audio = resnet18('audio')
        self.visual = resnet18('visual')
        self.txt_enc = BertClf(args)

        self.W_Q = nn.Linear(args.feature_dim, args.feature_dim)
        self.W_K = nn.Linear(args.feature_dim, args.feature_dim)
        self.W_V = nn.Linear(args.feature_dim, args.feature_dim)

        self.feature_extractor = nn.Sequential(
            nn.Linear(args.feature_dim * 2, args.feature_dim),
            nn.ReLU(),
            nn.Linear(args.feature_dim, args.feature_dim)
        )

        self.causal_classifier = nn.Linear(args.feature_dim, args.n_classes)

        self.lambda_v = args.lambda_v
        self.lambda_fe = args.lambda_fe

        self.feature_selection = nn.Sequential(
            nn.Linear(args.feature_dim, args.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(args.feature_dim // 2, args.feature_dim),
            nn.Sigmoid()
        )

        self.attention_scale = nn.Parameter(torch.ones(args.feature_dim))
        self.W_causal = nn.Parameter(torch.ones(args.feature_dim, args.feature_dim))

    def forward(self, txt, mask_t, segment_t, audio, visual, y_true):
        audio = audio.unsqueeze(1)
        a = self.audio(audio)
        v = self.visual(visual)
        t_hidden, _ = self.txt_enc(txt, mask_t, segment_t)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v.view(a.size(0), -1, *v.shape[1:]).permute(0, 2, 1, 3, 4), 1)
        a_out = torch.flatten(a, 1)
        v_out = torch.flatten(v, 1)

        t_out = t_hidden * self.feature_selection(t_hidden)
        a_out = a_out * self.feature_selection(a_out)

        multimodal_features = torch.cat([t_out.unsqueeze(1), a_out.unsqueeze(1)], dim=1)

        Q = self.W_Q(multimodal_features)
        K = self.W_K(multimodal_features)
        V = self.W_V(multimodal_features)

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.args.feature_dim ** 0.5)
        attention_scores = attention_scores * self.attention_scale.unsqueeze(0).unsqueeze(0)
        attention_weights = F.softmax(attention_scores, dim=-1)
        V = torch.bmm(attention_weights, V).sum(dim=1)

        z_c = self.feature_extractor(torch.cat([t_out, a_out], dim=1))
        z_c.requires_grad_(True)
        delta = -torch.autograd.grad(
            outputs=self.causal_classifier(z_c).log_softmax(dim=1),
            inputs=z_c,
            grad_outputs=torch.ones_like(z_c),
            retain_graph=True,
            create_graph=True
        )[0]
        z_c_bar = z_c + delta

        kl_divergence = F.kl_div(z_c_bar.log_softmax(dim=1), z_c.softmax(dim=1), reduction='batchmean')

        y_pred_real = self.causal_classifier(z_c).argmax(dim=1)
        y_pred_counterfactual = self.causal_classifier(z_c_bar).argmax(dim=1)

        sufficiency_risk = (y_pred_real != y_true).float().mean()
        necessity_risk = (y_pred_counterfactual == y_true).float().mean()
        c3_risk = sufficiency_risk + necessity_risk

        loss_v = kl_divergence
        loss_fe = F.kl_div(z_c_bar.log_softmax(dim=1), z_c.softmax(dim=1), reduction='batchmean')
        total_loss = c3_risk + self.lambda_v * loss_v + self.lambda_fe * loss_fe

        return total_loss, c3_risk, loss_v, loss_fe