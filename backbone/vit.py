
from torchvision.models.vision_transformer import vit_b_16

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vit_b_16(pretrained=True)

    def forward(self, x):
        return self.backbone(x)