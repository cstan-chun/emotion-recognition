import os
import torch
import numpy as np
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", max_samples: int = None):
        self.samples = []
        self.emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        self.data_dir = data_dir
        self._load_split(split, max_samples)

    def _load_split(self, split: str, max_samples: int = None):
        split_dir = os.path.join(self.data_dir, split)
        if not os.path.exists(split_dir):
            return
        for emotion in self.emotions:
            emo_dir = os.path.join(split_dir, emotion)
            if not os.path.exists(emo_dir):
                continue
            for fname in os.listdir(emo_dir):
                if fname.endswith(".npz"):
                    self.samples.append({
                        "path": os.path.join(emo_dir, fname),
                        "emotion": self.emotions.index(emotion),
                    })
        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = np.load(sample["path"])
        mfcc = torch.from_numpy(data["mfcc"]).float()
        frames = torch.from_numpy(data["frames"]).float()
        label = sample["emotion"]
        return mfcc, frames, label


def collate_emotion_batch(batch: list) -> tuple:
    mfccs, frames, labels = zip(*batch)
    max_mfcc_len = max(m.shape[0] for m in mfccs)
    padded_mfccs = []
    for m in mfccs:
        if m.shape[0] < max_mfcc_len:
            pad = torch.zeros(max_mfcc_len - m.shape[0], m.shape[1])
            m = torch.cat([m, pad], dim=0)
        padded_mfccs.append(m)
    return (
        torch.stack(padded_mfccs),
        torch.stack(frames),
        torch.tensor(labels),
    )
