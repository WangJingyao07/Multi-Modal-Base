import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-2])

    def forward(self, x):
        return self.backbone(x)

class ResNetClf(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        logits = self.classifier(x)
        return x, logits