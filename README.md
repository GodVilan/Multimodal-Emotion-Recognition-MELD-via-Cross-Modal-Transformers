# 🎭 Multimodal Emotion Recognition (MELD) via Cross-Modal Transformers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Transformers-orange)](https://huggingface.co/models)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, production-grade multimodal deep learning pipeline designed for Conversational Emotion Recognition (ERC). This system integrates **Linguistic (BERT)**, **Acoustic (Wav2Vec2)**, and **Visual (ViT)** signals using a custom cross-modal fusion architecture to classify emotions in multi-party dialogues.

---

## 🚀 Executive Summary & Key Results

Following 5 epochs of staged fine-tuning on the highly imbalanced MELD dataset, the system achieved the following metrics on the unseen test set:

| Metric | Score | Performance Context |
| :--- | :--- | :--- |
| **Accuracy** | **59.96%** | Highly competitive baseline for 7-class imbalanced real-world data. |
| **Weighted F1** | **60.66%** | Demonstrates robust precision/recall balance across primary classes. |
| **Macro F1** | **44.72%** | Significantly improved via dynamic Class-Weighted Loss optimization. |

---

## 🧠 System Architecture

The model implements a **Late Fusion Transformer** approach. Instead of simple vector concatenation, it projects modality-specific features into a shared embedding space ($d=768$) and utilizes a Transformer Encoder to learn complex inter-modality correlations.

### 🔹 Foundation Encoders
* **Text:** `bert-base-uncased` — Extracts semantic and contextual linguistic features.
* **Audio:** `facebook/wav2vec2-base` — Extracts acoustic properties and prosody from raw 16kHz waveforms.
* **Vision:** `google/vit-base-patch16-224` — Processes sampled video frames via Vision Transformer spatiotemporal patches.

### 🔹 Fusion Logic
The system uses a **4-layer Transformer Encoder** (8 attention heads) to perform cross-modal reasoning. This allows the network to dynamically weigh, for example, the importance of a specific spoken word against a micro-expression or vocal tone. The output is mean-pooled and passed through an MLP classifier with Dropout ($0.3$).

---

## 🔥 Engineering Highlights

* **Mixed Precision Training (AMP):** Utilizes `torch.amp.GradScaler` to optimize GPU memory footprint and increase throughput, critical for training large multimodal networks on consumer-grade hardware.
* **Staged Fine-Tuning Strategy:** To stabilize training, the system uses a 2-epoch "Encoder Freeze." The cross-modal fusion head is warmed up first, followed by full-model unfreezing to refine the foundation backbone weights without catastrophic forgetting.
* **Cached Tensor Preprocessing:** Decouples heavy media decoding (OpenCV/Librosa) from the training loop. Converting raw video/audio into serialized `.pt` files results in near-zero data-loading bottlenecks during GPU execution.
* **Class Imbalance Mitigation:** Automatically computes class weights from the training distribution to scale the `CrossEntropyLoss`, ensuring minority target variables (e.g., Disgust, Fear) receive proportional gradient updates.

---

## 🏗 Project Structure

```text
multimodal-emotion-recognition/
│
├── src/
│   ├── dataset.py       # PyTorch Dataset for loading .pt cached tensors
│   ├── encoders.py      # Wrapper classes for BERT, Wav2Vec2, and ViT backbones
│   ├── fusion.py        # Transformer-based Cross-Modal attention logic
│   ├── model.py         # Main MultiModalModel with modality ablation switches
│   ├── train.py         # AMP-enabled training epoch logic
│   ├── evaluate.py      # Metrics suite (Accuracy, F1, Confusion Matrix)
│   └── utils.py         # CV2/Librosa helpers for frame & audio extraction
│
├── data/
│   └── MELD/ (Not Tracked)
│
├── config.py            # Global hyperparameters & device configuration
├── preprocess.py        # Multi-threaded feature extraction pipeline
└── main.py              # Entry point for training and final evaluation
```

---

## ⚙️ Installation & Usage

### 1. Requirements
Ensure you are running Python 3.10+ and have a CUDA-enabled GPU.
```bash
pip install torch torchvision torchaudio transformers datasets librosa opencv-python scikit-learn seaborn matplotlib
```

### 2. Preprocessing
Before training, run the preprocessing script to extract and cache all modality features. **This only needs to be run once.**
```bash
python preprocess.py
```

### 3. Training & Evaluation
Execute the main script. This handles the staged unfreezing, AMP training loop, model checkpointing, and final test set evaluation.
```bash
python main.py
```

---

## 🧪 Ablation Study Support
The architecture is designed for modularity. You can toggle modalities in `model.py` to validate the effectiveness of the cross-modal fusion:
```python
# Example: Audio-Visual only (Ablating Text)
model = MultiModalModel(use_text=False, use_audio=True, use_vision=True)
```

---

## 👤 Author & Contact

Developed as a demonstration of production-grade Machine Learning architecture, multimodal system design, and large-model fine-tuning. 

**Availability Note:** Actively seeking full-time opportunities in Data Science, Machine Learning, and Gen AI Engineering in the United States, available to begin in June 2026.