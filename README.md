# 情绪识别系统 V2.0

基于视频的**视听多模态情绪识别系统**。从视频中检测并追踪所有人脸，对说话人进行多模态情绪识别（音频+视觉），对非说话人进行纯视觉情绪识别，输出标注视频和结构化 JSON 结果。

## 应用场景

离线批处理短视频（1-3 人，脸部较清晰），准确率优先。

## 情绪类别

7 类：`angry` `disgust` `fear` `happy` `neutral` `sad` `surprise`

## 系统架构

```
视频输入
  ├── SCRFD 人脸检测（脸框+关键点）
  ├── ByteTrack 多目标追踪（分配 person_id）
  └── Silero VAD 语音检测（说话起止时间段）
              │
              ▼
  ┌──────────────────────────────────────────┐
  │              说话人匹配                    │
  │      (唇动 × 音频能量 皮尔逊相关系数)        │
  └──────────────────────────────────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
  说话人             非说话人
  路径A              路径B
  多模态融合           纯视觉 1s 窗口
  ┌─────────────┐    ┌──────────────┐
  │MFCC → 1D CNN│    │抽15帧 → EffNet│
  │抽15帧→EffNet│    │Mean Pool     │
  │Cross-Attn   │    │FC → 7类       │
  │FC → 7类      │    └──────────────┘
  └─────────────┘
              │
              ▼
  标注视频 + 结构化 JSON 输出
```

- **感知层**：使用预训练模型（不训练），每帧运行
- **情绪层**：PyTorch 模型（需训练），根据 VAD 结果走双路径
- 两条路径共享 EfficientNet-B0 视觉骨干

## 项目结构

```
├── configs/default.yaml      # 模型、训练、推理配置
├── src/
│   ├── config.py             # YAML 配置加载器
│   ├── perception/
│   │   ├── detector.py       # SCRFD 人脸检测
│   │   ├── tracker.py        # ByteTrack 多目标追踪
│   │   └── vad.py            # Silero VAD 语音活动检测
│   ├── emotion/
│   │   ├── audio_encoder.py  # MFCC → 1D-CNN → [15, 256]
│   │   ├── visual_encoder.py # EfficientNet-B0 → [15, 256]
│   │   ├── fusion.py         # Cross-Attention 融合
│   │   ├── classifier.py     # FC → 7 类
│   │   └── model.py          # 组合模型（3种前向路径）
│   ├── speaker.py            # 唇动相关性说话人匹配
│   ├── pipeline.py           # 推理管道编排
│   ├── output.py             # 标注视频 + JSON 生成
│   └── data_utils.py         # 数据集 + batch 整理
├── scripts/
│   ├── train.py              # 训练入口
│   └── inference.py          # 推理入口
├── tests/                    # pytest 单元测试
├── data/raw/                 # 原始数据集
├── data/processed/           # 预处理后的 .npz 文件
├── checkpoints/              # 模型权重
└── outputs/                  # 推理输出
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

将以下数据集放入 `data/raw/`：

| 数据集 | 模态 | 用途 |
|--------|------|------|
| [RAVDESS](https://zenodo.org/record/1188976) | 音频+视频 | 单模态预训练 |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | 音频+视频 | 单模态预训练 |
| [IEMOCAP](https://sail.usc.edu/iemocap/) | 音频+视频 | 多模态联合训练 |
| [MELD](https://github.com/declare-lab/MELD) | 音频+视频 | 多模态联合训练/评测 |

### 3. 预处理数据

将原始数据集转换为统一 `.npz` 格式（以 RAVDESS 为例）：

```python
python scripts/preprocess_ravdess.py
```

预处理后的目录结构：
```
data/processed/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
├── val/
└── test/
```

### 4. 训练模型

三阶段渐进训练，按顺序执行（每阶段耗时因数据量而异，建议 GPU）：

```bash
# 阶段1: 视觉编码器预训练（EfficientNet-B0）
python scripts/train.py --stage visual --epochs 30 --lr 0.001 --data_dir data/processed

# 阶段1: 音频编码器预训练（MFCC + 1D-CNN）
python scripts/train.py --stage audio --epochs 30 --lr 0.001 --data_dir data/processed

# 阶段2: 多模态联合训练（冻结编码器底层，训练融合+分类头）
python scripts/train.py --stage joint --epochs 20 --lr 0.0005 --data_dir data/processed

# 阶段3: 端到端微调
python scripts/train.py --stage finetune --epochs 10 --lr 0.0001 --data_dir data/processed
```

### 5. 推理

```bash
python scripts/inference.py \
  --video path/to/video.mp4 \
  --checkpoint checkpoints/best_finetune.pt \
  --output_dir outputs/
```

输出：
- `outputs/annotated_output.mp4` — 标注视频（说话人绿框，非说话人灰框）
- `outputs/result.json` — 结构化推理结果

## 运行测试

```bash
# 全部测试
pytest tests/ -v

# 单个测试文件
pytest tests/test_tracker.py -v
```

注意：`test_detector.py` 和 `test_visual_encoder.py` 需要网络下载预训练模型，网络受限时会自动跳过。

## 配置

编辑 `configs/default.yaml` 调整模型结构、训练超参、感知层阈值等。Python 中通过点号访问：

```python
from config import load_config
cfg = load_config()
print(cfg.model.visual.embedding_dim)  # 256
print(cfg.training.lr_pretrain)        # 0.001
```

## 输出格式

### 标注视频

| 状态 | 框颜色 | 标签内容 |
|------|--------|---------|
| 正在说话 | 绿色 | 情绪 + 置信度 |
| 未说话 | 灰色 | Person ID + 情绪 |

### JSON

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
  }]
}
```

## 技术栈

PyTororch · SCRFD (insightface) · ByteTrack · Silero VAD · EfficientNet-B0 · librosa · OpenCV
