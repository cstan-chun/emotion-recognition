import torch
import torch.nn as nn
from .audio_encoder import AudioEncoder
from .visual_encoder import VisualEncoder
from .fusion import CrossAttentionFusion
from .classifier import EmotionClassifier


class EmotionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        v_cfg = config.model.visual
        a_cfg = config.model.audio
        f_cfg = config.model.fusion
        c_cfg = config.model.classifier

        self.audio_encoder = AudioEncoder(
            mfcc_dim=a_cfg.mfcc_dim,
            embedding_dim=a_cfg.embedding_dim,
            num_timesteps=a_cfg.num_timesteps,
        )
        self.visual_encoder = VisualEncoder(
            backbone=v_cfg.backbone,
            pretrained=v_cfg.pretrained,
            embedding_dim=v_cfg.embedding_dim,
        )
        self.fusion = CrossAttentionFusion(
            dim=f_cfg.dim,
            num_heads=f_cfg.num_heads,
        )
        self.classifier = EmotionClassifier(
            input_dim=f_cfg.dim,
            hidden_dim=c_cfg.hidden_dim,
            num_classes=c_cfg.num_classes,
        )
        self.visual_classifier = EmotionClassifier(
            input_dim=v_cfg.embedding_dim,
            hidden_dim=c_cfg.hidden_dim,
            num_classes=c_cfg.num_classes,
        )
        self.audio_classifier = EmotionClassifier(
            input_dim=a_cfg.embedding_dim,
            hidden_dim=c_cfg.hidden_dim,
            num_classes=c_cfg.num_classes,
        )

    def forward_multimodal(self, mfcc: torch.Tensor,
                           frames: torch.Tensor) -> torch.Tensor:
        audio_feat = self.audio_encoder(mfcc)
        visual_feat = self.visual_encoder(frames)
        fused = self.fusion(visual_feat, audio_feat)
        return self.classifier(fused)

    def forward_visual_only(self, frames: torch.Tensor) -> torch.Tensor:
        visual_feat = self.visual_encoder(frames)
        pooled = visual_feat.mean(dim=1)
        return self.visual_classifier(pooled)

    def forward_audio_only(self, mfcc: torch.Tensor) -> torch.Tensor:
        audio_feat = self.audio_encoder(mfcc)
        pooled = audio_feat.mean(dim=1)
        return self.audio_classifier(pooled)
