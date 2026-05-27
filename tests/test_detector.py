import pytest
import numpy as np

from perception.detector import FaceDetector


def test_detector_empty_frame():
    det = FaceDetector()
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(dummy)
        assert isinstance(results, list)
    except OSError:
        pytest.skip("Skipping: unable to download SCRFD model")


def test_detector_output_format():
    det = FaceDetector()
    try:
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = det.detect(dummy)
        for r in results:
            assert "bbox" in r
            assert len(r["bbox"]) == 4
            assert "confidence" in r
    except OSError:
        pytest.skip("Skipping: unable to download SCRFD model")
