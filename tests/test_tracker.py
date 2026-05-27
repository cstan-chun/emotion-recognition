
from perception.tracker import FaceTracker


def test_tracker_new_detection():
    tracker = FaceTracker()
    dets = [{"bbox": [100, 100, 50, 60], "confidence": 0.9}]
    results = tracker.update(dets)
    assert len(results) == 1
    assert results[0]["person_id"] == 0


def test_tracker_persistent_id():
    tracker = FaceTracker()
    dets1 = [{"bbox": [100, 100, 50, 60], "confidence": 0.9}]
    r1 = tracker.update(dets1)
    dets2 = [{"bbox": [102, 101, 50, 60], "confidence": 0.9}]
    r2 = tracker.update(dets2)
    assert r2[0]["person_id"] == r1[0]["person_id"]


def test_tracker_reset():
    tracker = FaceTracker()
    tracker.update([{"bbox": [100, 100, 50, 60], "confidence": 0.9}])
    tracker.reset()
    assert len(tracker.tracks) == 0
