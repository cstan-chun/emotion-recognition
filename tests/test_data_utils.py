import torch
import numpy as np
import os
import tempfile

from data_utils import EmotionDataset, collate_emotion_batch


def test_dataset_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "train", "happy"))
        mfcc = np.random.randn(100, 40).astype(np.float32)
        frames = np.random.randn(15, 3, 224, 224).astype(np.float32)
        np.savez(os.path.join(tmpdir, "train", "happy", "sample.npz"),
                 mfcc=mfcc, frames=frames)
        ds = EmotionDataset(tmpdir, split="train")
        assert len(ds) == 1
        m, f, l = ds[0]
        assert m.shape == (100, 40)
        assert f.shape == (15, 3, 224, 224)
        assert l == 3


def test_collate():
    samples = [
        (torch.randn(100, 40), torch.randn(15, 3, 224, 224), 0),
        (torch.randn(120, 40), torch.randn(15, 3, 224, 224), 1),
    ]
    mfccs, frames, labels = collate_emotion_batch(samples)
    assert mfccs.shape == (2, 120, 40)
    assert frames.shape == (2, 15, 3, 224, 224)
    assert labels.shape == (2,)
