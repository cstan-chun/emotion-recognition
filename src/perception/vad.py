import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps


class VoiceActivityDetector:
    def __init__(self, threshold: float = 0.5,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 300):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = load_silero_vad()

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
        self._load_model()

        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        audio_t = torch.from_numpy(audio).float()
        speech_timestamps = get_speech_timestamps(
            audio_t, self._model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=True,
        )

        return [(ts["start"], ts["end"]) for ts in speech_timestamps]
