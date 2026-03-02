import cv2
import torch
import numpy as np
import librosa
from config import NUM_FRAMES, AUDIO_MAX_LEN

def sample_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If video is broken or empty
    if total_frames <= 0:
        cap.release()
        return torch.zeros((NUM_FRAMES, 3, 224, 224))

    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()

    # If reading failed partially
    if len(frames) == 0:
        return torch.zeros((NUM_FRAMES, 3, 224, 224))

    # If fewer frames than expected, pad
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])

    frames = np.array(frames)
    frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0

    return frames

def extract_audio(video_path):
    try:
        audio, _ = librosa.load(video_path, sr=16000)
    except Exception:
        # If file is corrupted or unreadable
        return torch.zeros(AUDIO_MAX_LEN)

    if len(audio) > AUDIO_MAX_LEN:
        audio = audio[:AUDIO_MAX_LEN]
    else:
        padding = AUDIO_MAX_LEN - len(audio)
        audio = np.pad(audio, (0, padding))

    return torch.tensor(audio)
