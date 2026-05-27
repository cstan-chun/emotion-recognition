import numpy as np
import cv2
from scipy.stats import pearsonr

MOUTH_LANDMARK_INDICES = list(range(48, 68))


def extract_mouth_roi(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    mouth_pts = landmarks[MOUTH_LANDMARK_INDICES].astype(int)
    x_min = max(np.min(mouth_pts[:, 0]) - 5, 0)
    y_min = max(np.min(mouth_pts[:, 1]) - 5, 0)
    x_max = min(np.max(mouth_pts[:, 0]) + 5, frame.shape[1])
    y_max = min(np.max(mouth_pts[:, 1]) + 5, frame.shape[0])
    if x_max <= x_min or y_max <= y_min:
        return np.zeros((10, 10), dtype=np.uint8)
    roi = frame[y_min:y_max, x_min:x_max]
    return cv2.resize(roi, (32, 32))


def compute_lip_motion_series(frames: list[np.ndarray],
                              landmarks_list: list[np.ndarray]) -> np.ndarray:
    motions = []
    prev_roi = None
    for frame, landmarks in zip(frames, landmarks_list):
        if landmarks is None:
            motions.append(0.0)
            prev_roi = None
            continue
        roi = extract_mouth_roi(frame, landmarks)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        if prev_roi is not None:
            diff = np.mean(np.abs(roi_gray.astype(float) -
                                  prev_roi.astype(float)))
            motions.append(diff)
        else:
            motions.append(0.0)
        prev_roi = roi_gray
    return np.array(motions)


def compute_audio_rms(audio: np.ndarray, hop_len: int = 160,
                      num_frames: int = None) -> np.ndarray:
    if num_frames is None:
        num_frames = len(audio) // hop_len
    rms = []
    for i in range(num_frames):
        start = i * hop_len
        chunk = audio[start:start + hop_len]
        rms.append(np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0)
    return np.array(rms)


def match_speaker(person_frames: dict[int, list],
                  person_landmarks: dict[int, list],
                  audio: np.ndarray,
                  speech_segment: tuple[float, float],
                  sample_rate: int = 16000,
                  video_fps: float = 30.0) -> int:
    start_sample = int(speech_segment[0] * sample_rate)
    end_sample = int(speech_segment[1] * sample_rate)
    speech_audio = audio[start_sample:end_sample]

    hop_len = int(sample_rate * 0.01)
    num_rms = max(1, len(speech_audio) // hop_len)

    rms = compute_audio_rms(speech_audio, hop_len=hop_len, num_frames=num_rms)

    best_corr = -1.0
    best_pid = -1

    for pid in person_frames:
        frames_sublist = person_frames[pid]
        landmarks_sublist = person_landmarks.get(pid, [None] * len(frames_sublist))
        lip_motion = compute_lip_motion_series(frames_sublist, landmarks_sublist)

        target_len = min(len(lip_motion), len(rms))
        if target_len < 5:
            continue

        lip_motion = lip_motion[:target_len]
        rms_cut = rms[:target_len]

        if np.std(lip_motion) < 1e-6 or np.std(rms_cut) < 1e-6:
            continue

        corr, _ = pearsonr(lip_motion, rms_cut)
        if np.isnan(corr):
            continue
        if corr > best_corr:
            best_corr = corr
            best_pid = pid

    return best_pid
