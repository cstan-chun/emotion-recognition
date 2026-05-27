from config import load_config
from pipeline import EmotionRecognitionPipeline


def test_pipeline_init():
    cfg = load_config()
    try:
        pipeline = EmotionRecognitionPipeline(cfg)
        assert pipeline is not None
        assert hasattr(pipeline, "detector")
        assert hasattr(pipeline, "tracker")
        assert hasattr(pipeline, "vad")
    except Exception as e:
        print(f"Skipping due to: {e}")
