from setuptools import setup, find_packages

setup(
    name="emotion_recognition",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    py_modules=["config", "data_utils", "output", "pipeline", "speaker"],
)
