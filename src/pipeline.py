import os
import json
import numpy as np
import cv2
import librosa
import torch
from perception.detector import FaceDetector
from perception.tracker import FaceTracker
from perception.vad import VoiceActivityDetector
from emotion.model import EmotionModel
from emotion.audio_encoder import AudioEncoder
from speaker import match_speaker


class EmotionRecognitionPipeline:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = FaceDetector(
            conf_thresh=config.perception.detector.confidence_threshold,
        )
        self.tracker = FaceTracker(
            track_thresh=config.perception.tracker.track_thresh,
            match_thresh=config.perception.tracker.match_thresh,
            track_buffer=config.perception.tracker.track_buffer,
        )
        self.vad = VoiceActivityDetector(
            threshold=config.perception.vad.threshold,
            min_speech_duration_ms=config.perception.vad.min_speech_duration_ms,
            min_silence_duration_ms=config.perception.vad.min_silence_duration_ms,
        )
        self.model = EmotionModel(config).to(self.device)
        self.model.eval()
        self.labels = config.model.classifier.emotion_labels

    def process(self, video_path: str, checkpoint_path: str = None) -> dict:
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        all_detections = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            dets = self.detector.detect(frame)
            tracked = self.tracker.update(dets)
            all_detections.append(tracked)
            frame_idx += 1
        cap.release()
        duration = total_frames / fps if fps > 0 else 0

        audio, sr = self._extract_audio(video_path)
        speech_segments = self.vad.detect(audio, sr) if audio is not None else []

        person_tracks = self._build_tracks(all_detections)
        speech_results = []

        for seg in speech_segments:
            speaker_id = self._identify_speaker(
                person_tracks, frames, all_detections, audio, seg, fps, sr,
            )
            if speaker_id >= 0:
                emotion = self._predict_multimodal(
                    audio, sr, frames, all_detections, speaker_id, seg, fps,
                )
                speech_results.append({
                    "segment_id": len(speech_results),
                    "start_sec": seg[0],
                    "end_sec": seg[1],
                    "person_id": speaker_id,
                    "modality": "audio_visual",
                    "emotion": emotion,
                })

        active_speakers = {s["person_id"] for s in speech_results}
        visual_results = self._predict_visual_sliding(
            frames, all_detections, person_tracks, active_speakers, fps, duration,
        )

        return {
            "video": {
                "path": video_path,
                "duration_sec": duration,
                "fps": fps,
                "resolution": f"{width}x{height}",
            },
            "tracks": person_tracks,
            "speech_segments": speech_results,
            "visual_segments": visual_results,
            "summary": {
                "total_persons": len(person_tracks),
                "total_speech_segments": len(speech_results),
            },
        }

    def _extract_audio(self, video_path: str):
        try:
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-vn",
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                tmp_name,
            ], capture_output=True, check=True)
            audio, sr = librosa.load(tmp_name, sr=16000)
            os.unlink(tmp_name)
            return audio, sr
        except Exception:
            return np.zeros(16000, dtype=np.float32), 16000

    def _build_tracks(self, all_detections: list) -> list:
        pid_to_frames = {}
        for fi, dets in enumerate(all_detections):
            for d in dets:
                pid = d["person_id"]
                if pid not in pid_to_frames:
                    pid_to_frames[pid] = {"person_id": pid, "face_bboxes": [],
                                          "appear_frames": []}
                pid_to_frames[pid]["face_bboxes"].append({
                    "frame": fi,
                    "x": int(d["bbox"][0]),
                    "y": int(d["bbox"][1]),
                    "w": int(d["bbox"][2]),
                    "h": int(d["bbox"][3]),
                })
                pid_to_frames[pid]["appear_frames"].append(fi)
        for pid in pid_to_frames:
            frames_list = pid_to_frames[pid]["appear_frames"]
            pid_to_frames[pid]["appear_frames"] = [min(frames_list), max(frames_list)]
        return list(pid_to_frames.values())

    def _identify_speaker(self, person_tracks, frames, detections,
                          audio, seg, fps, sr):
        start_f = int(seg[0] * fps)
        end_f = int(seg[1] * fps)
        person_frames = {}
        person_landmarks = {}
        for pid_data in person_tracks:
            pid = pid_data["person_id"]
            person_frames[pid] = []
            person_landmarks[pid] = []
            for offset, frame_dets in enumerate(detections[start_f:end_f + 1]):
                fi = start_f + offset
                for d in frame_dets:
                    if d["person_id"] == pid:
                        person_frames[pid].append(frames[fi])
                        person_landmarks[pid].append(d.get("landmarks"))
                        break
        return match_speaker(person_frames, person_landmarks, audio, seg, sr, fps)

    def _predict_multimodal(self, audio, sr, frames, detections,
                            speaker_id, seg, fps):
        start_f = int(seg[0] * fps)
        end_f = int(seg[1] * fps)
        seg_frames = list(range(start_f, min(end_f + 1, len(frames))))

        indices = np.linspace(0, len(seg_frames) - 1, 15, dtype=int)
        sampled = []
        for idx in indices:
            fi = seg_frames[idx]
            frame = frames[fi]
            face = self._get_face(frame, detections[fi], speaker_id)
            sampled.append(face)
        frames_t = torch.from_numpy(np.stack(sampled)).float().to(self.device)
        frames_t = frames_t.unsqueeze(0)

        start_s = int(seg[0] * sr)
        end_s = int(seg[1] * sr)
        audio_seg = audio[start_s:end_s]
        mfcc = AudioEncoder.compute_mfcc(audio_seg, sr)
        mfcc_t = torch.from_numpy(mfcc).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model.forward_multimodal(mfcc_t, frames_t)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        return {
            "prediction": self.labels[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probs": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
        }

    def _predict_visual_sliding(self, frames, detections, person_tracks,
                                 active_speakers, fps, duration):
        results = []
        window_sec = 1.0
        step_sec = 1.0
        all_pids = {t["person_id"] for t in person_tracks}
        non_speakers = all_pids - active_speakers

        t = 0.0
        while t + window_sec <= duration:
            start_f = int(t * fps)
            end_f = int((t + window_sec) * fps)
            seg_frames = list(range(start_f, min(end_f, len(frames))))
            if len(seg_frames) < 5:
                t += step_sec
                continue

            for pid in non_speakers:
                indices = np.linspace(0, len(seg_frames) - 1, 15, dtype=int)
                sampled = []
                valid = True
                for idx in indices:
                    fi = seg_frames[idx]
                    face = self._get_face(frames[fi], detections[fi], pid)
                    if face is None:
                        valid = False
                        break
                    sampled.append(face)
                if not valid:
                    continue

                frames_t = torch.from_numpy(np.stack(sampled)).float().to(self.device)
                frames_t = frames_t.unsqueeze(0)

                with torch.no_grad():
                    logits = self.model.forward_visual_only(frames_t)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                pred_idx = int(np.argmax(probs))
                results.append({
                    "person_id": pid,
                    "window_sec": [t, t + window_sec],
                    "modality": "visual_only",
                    "emotion": {
                        "prediction": self.labels[pred_idx],
                        "confidence": float(probs[pred_idx]),
                        "probs": {self.labels[i]: float(probs[i])
                                  for i in range(len(self.labels))},
                    },
                })
            t += step_sec
        return results

    def _get_face(self, frame, detections_list, target_pid):
        for d in detections_list:
            if d["person_id"] == target_pid:
                x, y, w, h = d["bbox"]
                x, y, w, h = max(0, int(x)), max(0, int(y)), int(w), int(h)
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                face = face.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                face = (face - mean) / std
                return face.transpose(2, 0, 1)
        return None
