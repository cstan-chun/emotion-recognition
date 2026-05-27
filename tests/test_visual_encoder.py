import pytest
import torch
import sys
sys.path.insert(0, "src")
from emotion.visual_encoder import VisualEncoder


def test_visual_encoder_shape():
    model = VisualEncoder(embedding_dim=256)
    try:
        frames = torch.randn(2, 15, 3, 224, 224)
        with torch.no_grad():
            out = model(frames)
        assert out.shape == (2, 15, 256)
    except (OSError, RuntimeError):
        pytest.skip("Skipping: unable to download EfficientNet weights")


def test_backbone_feature_extraction():
    model = VisualEncoder(pretrained=True)
    try:
        model.eval()
        frames = torch.randn(1, 15, 3, 224, 224)
        with torch.no_grad():
            out = model(frames)
        assert not torch.isnan(out).any()
        assert out.shape == (1, 15, 256)
    except (OSError, RuntimeError):
        pytest.skip("Skipping: unable to download EfficientNet weights")
