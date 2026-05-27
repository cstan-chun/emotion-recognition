"""
MELD 数据集预处理脚本

MELD 原始结构（两种格式混合）:
    data/raw/MELD/
    ├── train.tar.gz         # 内含 train_sent_emo.csv + train_splits/*.mp4
    ├── dev_sent_emo.csv     # 独立的 CSV
    ├── dev.tar.gz           # 内含 dev_splits_complete/*.mp4
    ├── test_sent_emo.csv
    └── test.tar.gz

输出:
    data/processed/
    ├── train/{emotion}/*.npz
    ├── val/{emotion}/*.npz
    └── test/{emotion}/*.npz

用法:
    python scripts/preprocess_meld.py --data_dir data/raw/MELD --output_dir data/processed
"""

import os
import argparse
import tarfile
import subprocess
import tempfile
import glob
import numpy as np
import pandas as pd
import librosa
import cv2
from tqdm import tqdm

from emotion.audio_encoder import AudioEncoder

MELD_EMOTION_MAP = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

SPLIT_MAP = {
    "train": "train",
    "dev": "val",
    "test": "test",
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_timestamp(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts)


def extract_tar_smart(tar_path: str, extract_dir: str):
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"解压 {os.path.basename(tar_path)}"):
            out_path = os.path.join(extract_dir, member.name)
            if os.path.exists(out_path) and os.path.getsize(out_path) == member.size:
                continue
            tar.extract(member, extract_dir, filter="tar")


def find_csv(search_dir: str, split_name: str) -> str | None:
    """递归搜索 CSV 文件"""
    pattern = os.path.join(search_dir, "**", f"{split_name}_sent_emo.csv")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def find_mp4(search_dir: str, dialogue_id: int, utterance_id: int) -> str | None:
    """递归搜索 MP4 文件"""
    names = [
        f"dia{dialogue_id}_utt{utterance_id}.mp4",
        f"dia{dialogue_id:03d}_utt{utterance_id:03d}.mp4",
    ]
    for base in names:
        pattern = os.path.join(search_dir, "**", base)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def extract_audio(mp4_path: str, start_sec: float, end_sec: float,
                  sr: int = 16000) -> np.ndarray | None:
    duration = end_sec - start_sec
    if duration <= 0:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name

    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-ss", str(start_sec), "-t", str(duration),
            "-i", mp4_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sr), "-ac", "1", tmp_name,
        ], capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or not os.path.exists(tmp_name):
            return None
        audio, _ = librosa.load(tmp_name, sr=sr)
        if len(audio) < sr * 0.1:
            return None
        return audio.astype(np.float32)
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def extract_frames(mp4_path: str, start_sec: float, end_sec: float,
                   num_frames: int = 15) -> np.ndarray:
    frames = np.zeros((num_frames, 3, 224, 224), dtype=np.float32)

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        cap.release()
        return frames

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) - 1
    if end_frame <= start_frame:
        cap.release()
        return frames

    indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - MEAN.reshape(1, 1, 3)) / STD.reshape(1, 1, 3)
            frames[i] = frame.transpose(2, 0, 1)

    cap.release()
    return frames


def preprocess_meld(data_dir: str, output_dir: str, max_samples: int = None):
    for out_split in SPLIT_MAP.values():
        for emo in EMOTION_LABELS:
            os.makedirs(os.path.join(output_dir, out_split, emo), exist_ok=True)

    for csv_split, out_split in SPLIT_MAP.items():
        tar_path = os.path.join(data_dir, f"{csv_split}.tar.gz")
        extract_dir = os.path.join(data_dir, f"{csv_split}_extracted")

        # 如果有 tar.gz，解压
        if os.path.exists(tar_path):
            if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
                print(f"\n解压 {csv_split}.tar.gz ...")
                extract_tar_smart(tar_path, extract_dir)

        # 查找 CSV：优先在解压目录中找，其次在 data_dir 根目录找
        csv_path = find_csv(extract_dir, csv_split) if os.path.exists(extract_dir) else None
        if not csv_path:
            csv_path = os.path.join(data_dir, f"{csv_split}_sent_emo.csv")

        if not csv_path or not os.path.exists(csv_path):
            print(f"警告: 找不到 {csv_split} 的 CSV，跳过")
            continue

        print(f"CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"处理 {csv_split} 集: {len(df)} 条")

        # 确定 MP4 搜索根目录
        mp4_root = extract_dir if os.path.exists(extract_dir) else data_dir

        processed = 0
        skipped_empty_audio = 0
        skipped_missing_mp4 = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=csv_split):
            meld_emo = row.get("Emotion", "").strip().lower()
            target_emo = MELD_EMOTION_MAP.get(meld_emo)
            if target_emo is None:
                continue

            dialogue_id = int(row["Dialogue_ID"])
            utterance_id = int(row["Utterance_ID"])

            mp4_path = find_mp4(mp4_root, dialogue_id, utterance_id)
            if mp4_path is None:
                skipped_missing_mp4 += 1
                continue

            # MELD 的 MP4 是每个 utterance 的独立片段，直接从 0 开始取完整时长
            cap = cv2.VideoCapture(mp4_path)
            if not cap.isOpened():
                skipped_empty_audio += 1
                cap.release()
                continue
            mp4_fps = cap.get(cv2.CAP_PROP_FPS)
            mp4_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            clip_duration = mp4_frames / mp4_fps if mp4_fps > 0 else 2.0

            try:
                audio = extract_audio(mp4_path, 0.0, clip_duration)
                if audio is None:
                    skipped_empty_audio += 1
                    continue

                mfcc = AudioEncoder.compute_mfcc(audio, 16000)
                frames = extract_frames(mp4_path, 0.0, clip_duration)

            except Exception as e:
                continue

            out_dir = os.path.join(output_dir, out_split, target_emo)
            out_name = f"dia{dialogue_id}_utt{utterance_id}.npz"
            np.savez(os.path.join(out_dir, out_name), mfcc=mfcc, frames=frames)
            processed += 1

            if max_samples and processed >= max_samples:
                break

        print(f"  {csv_split} → {out_split}: 成功 {processed} 条"
              f" (缺音频: {skipped_empty_audio}, 缺MP4: {skipped_missing_mp4})")


def main():
    parser = argparse.ArgumentParser(description="MELD 数据集预处理")
    parser.add_argument("--data_dir", default="data/raw/MELD")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--max_samples", type=int, default=None, help="每 split 最大样本数")
    args = parser.parse_args()

    preprocess_meld(args.data_dir, args.output_dir, args.max_samples)
    print("\n预处理完成")


if __name__ == "__main__":
    main()
