import numpy as np


class FaceDetector:
    def __init__(self, conf_thresh: float = 0.5):
        self.conf_thresh = conf_thresh
        self._app = None

    def _load_model(self):
        if self._app is None:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=-1, det_thresh=self.conf_thresh)

    def detect(self, frame: np.ndarray) -> list[dict]:
        self._load_model()
        bboxes = []
        faces = self._app.get(frame)
        if faces is None:
            return bboxes
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            bboxes.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "landmarks": face.kps.astype(np.float32) if hasattr(face, "kps") else None,
                "confidence": float(face.det_score),
            })
        return bboxes
