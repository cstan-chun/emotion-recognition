import os
import json
import tempfile

from config import load_config
from output import OutputGenerator


def test_generate_json():
    cfg = load_config()
    gen = OutputGenerator(cfg)
    result = {"test": "data", "summary": {"total_persons": 1}}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = gen.generate_json(result, tmpdir)
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["test"] == "data"


def test_output_generator_colors():
    cfg = load_config()
    gen = OutputGenerator(cfg)
    assert gen.speaking_color == (0, 255, 0)
    assert gen.silent_color == (150, 150, 150)
