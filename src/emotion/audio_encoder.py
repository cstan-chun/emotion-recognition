import torch
import torch.nn as nn
import python_speech_features
import numpy as np
import librosa


class AudioEncoder(nn.Module):
    def __init__(self, mfcc_dim: int = 40, embedding_dim: int = 256,
                 num_timesteps: int = 15):
        super().__init__()
        self.mfcc_dim = mfcc_dim
        self.num_timesteps = num_timesteps

        self.conv = nn.Sequential(
            nn.Conv1d(mfcc_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, embedding_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(num_timesteps)

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        x = mfcc.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        return x.transpose(1, 2)

    @staticmethod
    def compute_mfcc(audio: np.ndarray, sample_rate: int,
                     winlen: float = 0.025, winstep: float = 0.01,
                     numcep: int = 14) -> np.ndarray:
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        mfcc_raw = python_speech_features.mfcc(
            audio, samplerate=sample_rate,
            winlen=winlen, winstep=winstep, numcep=numcep,
            nfilt=26, preemph=0.97, appendEnergy=True,
        )
        delta1 = python_speech_features.delta(mfcc_raw, 1)
        delta2 = python_speech_features.delta(mfcc_raw, 2)
        stacked = np.hstack([mfcc_raw, delta1, delta2])
        if stacked.shape[1] > 40:
            stacked = stacked[:, :40]
        if len(stacked) < 40:
            delta2_extra = np.zeros_like(delta2)
            stacked = np.hstack([mfcc_raw, delta1, delta2_extra])
            if stacked.shape[1] > 40:
                stacked = stacked[:, :40]
        return stacked.astype(np.float32)
