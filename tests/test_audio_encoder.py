import torch
import numpy as np
import sys
sys.path.insert(0, "src")
from emotion.audio_encoder import AudioEncoder


def test_audio_encoder_shape():
    model = AudioEncoder(mfcc_dim=40, embedding_dim=256, num_timesteps=15)
    mfcc = torch.randn(2, 200, 40)
    out = model(mfcc)
    assert out.shape == (2, 15, 256)


def test_compute_mfcc():
    audio = np.random.randn(16000 * 2).astype(np.float32)
    mfcc = AudioEncoder.compute_mfcc(audio, 16000)
    assert mfcc.ndim == 2
    assert mfcc.shape[1] == 40


def test_encoder_output_variance():
    model = AudioEncoder()
    model.eval()
    mfcc = torch.randn(4, 200, 40)
    with torch.no_grad():
        out = model(mfcc)
    assert not torch.isnan(out).any()
