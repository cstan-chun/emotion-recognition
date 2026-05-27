import yaml
from pathlib import Path
from typing import Any


class Config:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r") as f:
            self._data = yaml.safe_load(f)

    def __getattr__(self, key: str) -> Any:
        if key in self._data:
            val = self._data[key]
            if isinstance(val, dict):
                return _DictWrap(val)
            return val
        raise AttributeError(f"Config has no key: {key}")


class _DictWrap:
    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, key: str) -> Any:
        if key in self._d:
            val = self._d[key]
            if isinstance(val, dict):
                return _DictWrap(val)
            return val
        raise AttributeError(f"No key: {key}")


def load_config(path: str = "configs/default.yaml") -> Config:
    return Config(path)
