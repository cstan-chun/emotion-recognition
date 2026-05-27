import sys
sys.path.insert(0, "src")
from config import load_config


def test_load_config():
    cfg = load_config()
    assert cfg.model.visual.embedding_dim == 256
    assert cfg.model.classifier.num_classes == 7
    assert cfg.model.classifier.emotion_labels == ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    assert cfg.perception.vad.threshold == 0.5
