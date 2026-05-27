import numpy as np
import sys
sys.path.insert(0, "src")
from speaker import (
    match_speaker, compute_audio_rms,
    compute_lip_motion_series, extract_mouth_roi,
)


def test_audio_rms():
    audio = np.sin(np.linspace(0, 100 * np.pi, 16000)).astype(np.float32)
    rms = compute_audio_rms(audio, hop_len=160, num_frames=50)
    assert len(rms) == 50
    assert np.all(rms >= 0)


def test_match_speaker_returns_id():
    dummy_frames = {0: [np.zeros((64, 64, 3), dtype=np.uint8)] * 30}
    dummy_landmarks = {0: [np.random.randn(68, 2).astype(np.float32)] * 30}
    audio = np.sin(np.linspace(0, 100 * np.pi, 48000)).astype(np.float32)
    result = match_speaker(dummy_frames, dummy_landmarks, audio,
                           (0.0, 1.0))
    assert isinstance(result, int)
