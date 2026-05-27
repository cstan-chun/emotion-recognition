import numpy as np

from perception.vad import VoiceActivityDetector


def test_vad_silence():
    vad = VoiceActivityDetector()
    silence = np.zeros(16000 * 2, dtype=np.float32)
    segments = vad.detect(silence, 16000)
    assert segments == []


def test_vad_noisy_audio():
    vad = VoiceActivityDetector()
    noise = np.random.randn(16000 * 3).astype(np.float32) * 0.1
    segments = vad.detect(noise, 16000)
    assert isinstance(segments, list)
