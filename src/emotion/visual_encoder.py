import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    def __init__(self, backbone: str = "efficientnet-b0", pretrained: bool = True,
                 embedding_dim: int = 256):
        super().__init__()
        self._backbone_name = backbone
        self._pretrained = pretrained
        self._embedding_dim = embedding_dim
        self._initialized = False

    def _init_backbone(self):
        if self._initialized:
            return
        from efficientnet_pytorch import EfficientNet
        if self._pretrained:
            self.backbone = EfficientNet.from_pretrained(self._backbone_name)
        else:
            self.backbone = EfficientNet.from_name(self._backbone_name)
        backbone_dim = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, self._embedding_dim),
            nn.ReLU(),
        )
        self._initialized = True

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        self._init_backbone()
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        x = self.backbone(x)
        x = self.proj(x)
        x = x.view(B, T, -1)
        return x
