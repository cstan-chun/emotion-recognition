"""
MELD 数据集预处理脚本

将 MELD 原始数据转换为训练所需的 .npz 格式。

MELD 原始结构（假设）:
    data/raw/meld/
    ├── train_sent_emo.csv
    ├── dev_sent_emo.csv
    ├── test_sent_emo.csv
    ├── train/
    │   └── *.wav          # 训练集音频
    ├── dev/
    │   └── *.wav          # 验证集音频
    └── test/
        └── *.wav          # 测试集音频

输出:
    data/processed/
    ├── train/{emotion}/*.npz
    ├── val/{emotion}/*.npz
    └── test/{emotion}/*.npz

用法:
    python scripts/preprocess_meld.py --data_dir data/raw/meld --output_dir data/processed
"""

import os
import argparse
import numpy as np
import pandas as pd
import librosa
import cv2
from tqdm import tqdm


from emotion.audio_encoder import AudioEncoder

# MELD 情绪到项目标签的映射
MELD_EMOTION_MAP = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}

# 7 类情绪标签
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# split 名称映射: CSV 中的名称 → 输出目录名
SPLIT_MAP = {
    "train": "train",
    "dev": "val",
    "test": "test",
}

# ImageNet 归一化参数
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_meld(data_dir: str, output_dir: str, max_samples: int = None):
    for out_split in SPLIT_MAP.values():
        for emo in EMOTION_LABELS:
            os.makedirs(os.path.join(output_dir, out_split, emo), exist_ok=True)

    for csv_split, out_split in SPLIT_MAP.items():
        csv_path = os.path.join(data_dir, f"{csv_split}_sent_emo.csv")
        wav_dir = os.path.join(data_dir, csv_split)

        if not os.path.exists(csv_path):
            print(f"警告: CSV 文件不存在，跳过 {csv_split}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        print(f"\n处理 {csv_split} 集: {len(df)} 条")

        processed = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=csv_split):
            meld_emo = row.get("Emotion", "").lower()
            target_emo = MELD_EMOTION_MAP.get(meld_emo)
            if target_emo is None:
                continue

            utterance_id = row.get("Utterance_ID", "")
            # MELD 音频文件名格式: dia{dialogue_id}_utt{utterance_id}.wav
            dialogue_id = row.get("Dialogue_ID", "")
            if isinstance(utterance_id, (int, float)) and isinstance(dialogue_id, (int, float)):
                wav_name = f"dia{int(dialogue_id)}_utt{int(utterance_id)}.wav"
            else:
                # 如果 CSV 中已有 Sr No. 列作为索引
                sr_no = row.get("Sr No.", _)
                wav_name = f"dia{dialogue_id}_utt{utterance_id}.wav"

            wav_path = os.path.join(wav_dir, wav_name)

            if not os.path.exists(wav_path):
                # 尝试其他可能的文件名格式
                alt_name = f"dia{int(dialogue_id):03d}_utt{int(utterance_id):03d}.wav"
                alt_path = os.path.join(wav_dir, alt_name)
                if os.path.exists(alt_path):
                    wav_path = alt_path
                else:
                    continue

            try:
                # 加载音频并计算 MFCC
                audio, sr = librosa.load(wav_path, sr=16000)
                mfcc = AudioEncoder.compute_mfcc(audio, sr)

                # 尝试生成 15 帧（若 CSV 有起止时间且视频存在）
                frames = _extract_frames(data_dir, row, csv_split)

            except Exception as e:
                print(f"  跳过 {wav_name}: {e}")
                continue

            out_dir = os.path.join(output_dir, out_split, target_emo)
            out_name = os.path.splitext(wav_name)[0] + ".npz"
            np.savez(os.path.join(out_dir, out_name), mfcc=mfcc, frames=frames)
            processed += 1

            if max_samples and processed >= max_samples:
                break

        print(f"  {csv_split} → {out_split}: 成功 {processed} 条")


def _extract_frames(data_dir: str, row, split: str) -> np.ndarray:
    """从视频中抽取 15 帧。若无视频则返回空帧张量。"""
    frames = np.zeros((15, 3, 224, 224), dtype=np.float32)

    # 尝试找对应的视频
    season = row.get("Season", None)
    episode = row.get("Episode", None)
    start = row.get("StartTime", None)
    end = row.get("EndTime", None)

    if season is None or episode is None or start is None or end is None:
        return frames

    video_name = f"s{int(season):02d}_e{int(episode):02d}.mp4"
    video_dir = os.path.join(data_dir, "videos")
    video_path = os.path.join(video_dir, video_name)

    if not os.path.exists(video_path):
        return frames

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return frames

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0

    duration = end - start
    if duration <= 0:
        cap.release()
        return frames

    indices = np.linspace(start * fps, end * fps - 1, 15, dtype=int)

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


def main():
    parser = argparse.ArgumentParser(description="MELD 数据集预处理")
    parser.add_argument("--data_dir", default="data/raw/meld", help="MELD 原始数据目录")
    parser.add_argument("--output_dir", default="data/processed", help="预处理输出目录")
    parser.add_argument("--max_samples", type=int, default=None, help="每 split 最大样本数（调试用）")
    args = parser.parse_args()

    preprocess_meld(args.data_dir, args.output_dir, args.max_samples)
    print("\n预处理完成")


if __name__ == "__main__":
    main()
