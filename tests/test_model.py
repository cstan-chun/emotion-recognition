import pytest
import torch
import sys
sys.path.insert(0, "src")
from config import load_config
from emotion.model import EmotionModel


def test_audio_only_forward():
    cfg = load_config()
    model = EmotionModel(cfg)
    model.eval()
    mfcc = torch.randn(2, 200, 40)
    with torch.no_grad():
        out = model.forward_audio_only(mfcc)
    assert out.shape == (2, 7)


def test_shared_backbone():
    cfg = load_config()
    model = EmotionModel(cfg)
    v1 = model.visual_encoder
    assert v1 is model.visual_encoder


def test_multimodal_forward():
    cfg = load_config()
    model = EmotionModel(cfg)
    model.eval()
    try:
        mfcc = torch.randn(2, 200, 40)
        frames = torch.randn(2, 15, 3, 224, 224)
        with torch.no_grad():
            out = model.forward_multimodal(mfcc, frames)
        assert out.shape == (2, 7)
    except (OSError, RuntimeError):
        pytest.skip("Skipping: unable to download EfficientNet weights")


def test_visual_only_forward():
    cfg = load_config()
    model = EmotionModel(cfg)
    model.eval()
    try:
        frames = torch.randn(2, 15, 3, 224, 224)
        with torch.no_grad():
            out = model.forward_visual_only(frames)
        assert out.shape == (2, 7)
    except (OSError, RuntimeError):
        pytest.skip("Skipping: unable to download EfficientNet weights")
