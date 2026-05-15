# 实时目标检测 + 多目标追踪 + 视听情绪识别系统

## 概述

从视频中检测并追踪所有人脸，对说话人进行视听多模态情绪识别，对非说话人进行纯视觉情绪识别。输出标注视频和结构化 JSON 结果。

## 应用场景

- 离线批处理短视频（1-3人，脸部较清晰）
- 准确率优先，延迟要求宽松

## 情绪类别

7 类：开心 (happy)、悲伤 (sad)、愤怒 (angry)、恐惧 (fear)、惊讶 (surprise)、厌恶 (disgust)、中性 (neutral)

## 技术栈

- 框架：PyTorch
- 检测：SCRFD（预训练，不做训练）
- 追踪：ByteTrack（预训练，不做训练）
- VAD：Silero VAD（预训练，不做训练）

---

## 系统架构

双层架构：感知层 + 多模态情绪层。

### 第一层：感知层

每帧运行，使用成熟预训练模型：

```
视频输入 → SCRFD 人脸检测（脸框+关键点）
         → ByteTrack 多目标追踪（分配 person_id）
         → Silero VAD 语音检测（输出说话起止时间段）
```

### 第二层：多模态情绪识别层

根据 VAD 结果走双路径：

#### 路径 A：说话人（多模态融合）

1. **音频分支**：提取语音段音频 → MFCC 特征 `[num_frames, 40]` → 1D Conv + Pooling 下采样到 15 步 → `[15, D_audio]`
2. **视觉分支**：说话段内等间隔抽 15 帧 → 人脸对齐 → EfficientNet-B0 → 取最后一层特征 → `[15, D_visual]`
3. **融合**：Cross-Attention（视觉为 Query，音频为 Key/Value）→ Mean Pool → `[D]`
4. **分类**：FC(D→128) → ReLU → FC(128→7) → 7 类 Softmax

#### 路径 B：非说话人（纯视觉，1s 窗口）

1. 以 1s 为固定窗口，在时间轴上以 1s 步长滑动（无重叠）
2. 每窗口内均匀取 15 帧 → 人脸对齐 → EfficientNet-B0 → 取最后一层特征 → `[15, D_visual]`
3. **分类**：Temporal Mean Pool → FC(D→128) → ReLU → FC(128→7) → 7 类 Softmax

#### 路径 B 与路径 A 共享 EfficientNet-B0 骨干网络。

#### 触发逻辑

| 状态 | 说话人 | 非说话人 |
|------|--------|----------|
| 有人说话 | 路径 A（多模态） | 路径 B（纯视觉） |
| 无人说话 | - | 全部人路径 B（纯视觉） |

---

## 说话人匹配

唇动相关性方法：

1. 从 SCRFD 人脸关键点（68/106 点）提取嘴部 ROI
2. 在 VAD 语音段内计算嘴部 ROI 帧间像素差
3. 与音频 RMS 能量包络做皮尔逊相关系数
4. 相关系数最高的人判定为说话人

---

## 训练策略

### 数据集

| 数据集 | 模态 | 情绪类别 | 用途 |
|--------|------|---------|------|
| CREMA-D | 音+视 | 6类 | 单模态预训练 |
| RAVDESS | 音+视 | 8类（含中性） | 单模态预训练 |
| IEMOCAP | 音+视 | 6类 | 多模态联合训练 |
| MELD | 音+视 | 7类 | 多模态联合训练/评测 |

### 三阶段训练

1. **单模态预训练**：音频编码器和视觉编码器分别在 CREMA-D + RAVDESS 上预训练
2. **多模态联合训练**：冻结编码器底层，训练 Cross-Attention 融合 + 分类头（IEMOCAP + MELD）
3. **端到端微调**：解冻全部参数，低学习率微调

### 损失函数

- Focal Loss（处理类别不平衡）

---

## 评估指标

- 准确率 (Accuracy)
- Weighted F1-Score

---

## 输出格式

### 标注视频

| 条件 | 框颜色 | 标签内容 |
|------|--------|---------|
| 正在说话 | 绿色 | 情绪 + 置信度 |
| 未说话 | 灰色 | Person ID + 情绪（纯视觉） |
| 沉默间隙 | 灰色 | Person ID + 情绪（纯视觉） |

### JSON 结构

```json
{
  "video": { "path": "...", "duration_sec": 15.2, "fps": 30, "resolution": "1920x1080" },
  "tracks": [{ "person_id": 1, "face_bboxes": [...], "appear_frames": [0, 455] }],
  "speech_segments": [{
    "segment_id": 0, "start_sec": 2.3, "end_sec": 4.8, "person_id": 1,
    "modality": "audio_visual",
    "emotion": { "prediction": "happy", "confidence": 0.87, "probs": {...} }
  }],
  "visual_segments": [{
    "person_id": 2, "window_sec": [0.0, 1.0],
    "modality": "visual_only",
    "emotion": { "prediction": "neutral", "confidence": 0.65, "probs": {...} }
  }],
  "summary": { "total_persons": 2, "total_speech_segments": 3 }
}
```

---

## 项目结构

```
emotion recognitionV2.0/
├── data/
│   ├── raw/                    # RAVDESS, CREMA-D, IEMOCAP, MELD
│   └── processed/              # 预处理特征缓存
├── src/
│   ├── perception/
│   │   ├── detector.py         # SCRFD 人脸检测
│   │   ├── tracker.py          # ByteTrack 多目标追踪
│   │   └── vad.py              # Silero VAD
│   ├── emotion/
│   │   ├── audio_encoder.py    # MFCC → 1D-CNN + Pooling → [15,D]
│   │   ├── visual_encoder.py   # 抽帧 → EfficientNet → [15,D]
│   │   ├── fusion.py           # Cross-Attention 融合
│   │   └── classifier.py       # FC → 7类 Softmax
│   ├── speaker.py              # 说话人匹配（唇动相关性）
│   ├── pipeline.py             # 主流程编排
│   └── output.py               # 视频标注 + JSON 输出
├── configs/
│   └── default.yaml
├── scripts/
│   ├── train.py
│   └── inference.py
├── outputs/
└── requirements.txt
```

---

## 依赖清单

```
torch, torchvision, opencv-python, numpy, scipy,
librosa, python_speech_features,
facenet-pytorch, onnxruntime,  # SCRFD
filterpy,                        # ByteTrack
silero-vad,                      # VAD
efficientnet-pytorch,
scikit-learn,                    # 评估
pyyaml, tqdm
```
