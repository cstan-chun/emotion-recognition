import os
import json
import numpy as np
import cv2


class OutputGenerator:
    def __init__(self, config):
        self.cfg = config
        self.speaking_color = tuple(config.output.face_box_color_speaking)
        self.silent_color = tuple(config.output.face_box_color_silent)
        self.labels = config.model.classifier.emotion_labels

    def generate_video(self, video_path: str, result: dict,
                       output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(output_dir, "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        tracks_by_frame = {}
        for track in result["tracks"]:
            pid = track["person_id"]
            for bbox in track["face_bboxes"]:
                fi = bbox["frame"]
                if fi not in tracks_by_frame:
                    tracks_by_frame[fi] = []
                tracks_by_frame[fi].append({"person_id": pid, "bbox": bbox})

        speech_map = {}
        for seg in result.get("speech_segments", []):
            for t in np.arange(seg["start_sec"], seg["end_sec"], 1.0 / fps):
                speech_map[int(round(t * fps))] = {
                    "person_id": seg["person_id"],
                    "emotion": seg["emotion"]["prediction"],
                    "confidence": seg["emotion"]["confidence"],
                    "modality": "audio_visual",
                }

        visual_map = {}
        for seg in result.get("visual_segments", []):
            for t in np.arange(seg["window_sec"][0], seg["window_sec"][1], 1.0 / fps):
                visual_map[int(round(t * fps))] = {
                    "person_id": seg["person_id"],
                    "emotion": seg["emotion"]["prediction"],
                    "confidence": seg["emotion"]["confidence"],
                    "modality": "visual_only",
                }

        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in tracks_by_frame:
                for track in tracks_by_frame[frame_idx]:
                    bbox = track["bbox"]
                    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                    pid = track["person_id"]

                    is_speaker = (frame_idx in speech_map and
                                  speech_map[frame_idx]["person_id"] == pid)
                    if is_speaker:
                        color = self.speaking_color
                        info = speech_map[frame_idx]
                        label = f"ID:{pid} {info['emotion']}({info['confidence']:.2f})"
                    elif frame_idx in visual_map and visual_map[frame_idx]["person_id"] == pid:
                        color = self.silent_color
                        info = visual_map[frame_idx]
                        label = f"ID:{pid} {info['emotion']}({info['confidence']:.2f})"
                    else:
                        color = self.silent_color
                        label = f"ID:{pid}"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        return out_path

    def generate_json(self, result: dict, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return out_path

    def generate_all(self, video_path: str, result: dict,
                     output_dir: str) -> tuple[str, str]:
        video_out = self.generate_video(video_path, result, output_dir)
        json_out = self.generate_json(result, output_dir)
        return video_out, json_out
