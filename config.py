import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5

# Data
MAX_TEXT_LEN = 64
NUM_FRAMES = 8
AUDIO_MAX_LEN = 16000 * 6  # 6 seconds
EMBED_DIM = 768
NUM_CLASSES = 7

EMOTION_MAP = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "fear": 4,
    "disgust": 5,
    "surprise": 6
}
