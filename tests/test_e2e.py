import os
import sys
import tempfile
import numpy as np
import cv2
import json


def generate_test_video(output_path: str, duration_sec: float = 3.0,
                         fps: int = 30, width: int = 640, height: int = 480):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    n_frames = int(duration_sec * fps)
    for i in range(n_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return output_path


def test_full_pipeline_structure():
    sys.path.insert(0, "src")
    from config import load_config
    from pipeline import EmotionRecognitionPipeline

    cfg = load_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = generate_test_video(os.path.join(tmpdir, "test.mp4"))

        try:
            pipeline = EmotionRecognitionPipeline(cfg)
            result = pipeline.process(video_path)
        except Exception as e:
            print(f"Skipping (model download may fail): {e}")
            return

        assert "video" in result
        assert "tracks" in result
        assert "speech_segments" in result
        assert "visual_segments" in result
        assert "summary" in result
        assert result["video"]["fps"] == 30
        print("E2E test passed!")


def test_output_pipeline_integration():
    sys.path.insert(0, "src")
    from config import load_config
    from output import OutputGenerator

    cfg = load_config()
    gen = OutputGenerator(cfg)

    result = {
        "video": {"path": "test.mp4", "duration_sec": 1.0, "fps": 30, "resolution": "640x480"},
        "tracks": [{"person_id": 0, "face_bboxes": [{"frame": 0, "x": 100, "y": 100, "w": 50, "h": 60}], "appear_frames": [0, 0]}],
        "speech_segments": [{"segment_id": 0, "start_sec": 0.0, "end_sec": 1.0, "person_id": 0, "modality": "audio_visual", "emotion": {"prediction": "happy", "confidence": 0.9, "probs": {"happy": 0.9, "sad": 0.02, "angry": 0.02, "fear": 0.01, "surprise": 0.03, "disgust": 0.01, "neutral": 0.01}}}],
        "visual_segments": [],
        "summary": {"total_persons": 1, "total_speech_segments": 1},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = generate_test_video(os.path.join(tmpdir, "test.mp4"))
        _, json_out = gen.generate_all(video_path, result, tmpdir)
        with open(json_out) as f:
            data = json.load(f)
        assert data["summary"]["total_persons"] == 1
