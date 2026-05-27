import numpy as np


class FaceTracker:
    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8,
                 track_buffer: int = 30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.next_id = 0
        self.tracks: dict[int, dict] = {}
        self.lost_tracks: dict[int, dict] = {}

    def update(self, detections: list[dict]) -> list[dict]:
        matched = {}
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        if self.tracks and detections:
            ious = self._compute_iou_matrix(detections, self.tracks)
            for det_idx in range(len(detections)):
                for track_id in self.tracks:
                    if det_idx not in unmatched_dets or track_id not in unmatched_tracks:
                        continue
                    if ious[det_idx][track_id] > self.match_thresh:
                        matched[det_idx] = track_id
                        unmatched_dets.remove(det_idx)
                        unmatched_tracks.remove(track_id)

        for det_idx in unmatched_dets:
            if detections[det_idx]["confidence"] >= self.track_thresh:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    "bbox": detections[det_idx]["bbox"],
                    "landmarks": detections[det_idx].get("landmarks"),
                    "confidence": detections[det_idx]["confidence"],
                    "lost": 0,
                }
                matched[det_idx] = new_id

        for track_id in unmatched_tracks:
            self.tracks[track_id]["lost"] += 1
            self.lost_tracks[track_id] = self.tracks.pop(track_id)

        for det_idx, track_id in matched.items():
            self.tracks[track_id] = {
                "bbox": detections[det_idx]["bbox"],
                "landmarks": detections[det_idx].get("landmarks"),
                "confidence": detections[det_idx]["confidence"],
                "lost": 0,
            }

        for track_id in list(self.lost_tracks.keys()):
            if self.lost_tracks[track_id]["lost"] > self.track_buffer:
                del self.lost_tracks[track_id]

        results = []
        for track_id, track in {**self.tracks, **self.lost_tracks}.items():
            results.append({
                "person_id": track_id,
                "bbox": track["bbox"],
                "landmarks": track.get("landmarks"),
                "confidence": track["confidence"],
            })
        return results

    def _compute_iou_matrix(self, detections, tracks) -> dict:
        ious = {}
        for di, det in enumerate(detections):
            ious[di] = {}
            dx, dy, dw, dh = det["bbox"]
            d_area = dw * dh
            for tid, track in tracks.items():
                tx, ty, tw, th = track["bbox"]
                t_area = tw * th
                ix = max(dx, tx)
                iy = max(dy, ty)
                iw = min(dx + dw, tx + tw) - ix
                ih = min(dy + dh, ty + th) - iy
                if iw > 0 and ih > 0:
                    inter = iw * ih
                    iou = inter / (d_area + t_area - inter)
                else:
                    iou = 0.0
                ious[di][tid] = iou
        return ious

    def reset(self):
        self.tracks.clear()
        self.lost_tracks.clear()
        self.next_id = 0
