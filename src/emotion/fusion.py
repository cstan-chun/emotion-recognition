import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1),
        )

    def forward(self, visual_feat: torch.Tensor,
                audio_feat: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(visual_feat, audio_feat, audio_feat)
        x = self.norm1(visual_feat + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x.mean(dim=1)
