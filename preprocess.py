import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

from config import MAX_TEXT_LEN, EMOTION_MAP
from src.utils import sample_frames, extract_audio


def preprocess_split(csv_path, video_folder, save_folder):

    os.makedirs(save_folder, exist_ok=True)

    df = pd.read_csv(csv_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        save_path = os.path.join(save_folder, f"{idx}.pt")

        # Skip if already processed
        if os.path.exists(save_path):
            continue

        text = row["Utterance"]
        emotion = row["Emotion"]
        label = EMOTION_MAP[emotion]

        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt"
        )

        video_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        video_path = os.path.join(video_folder, video_name)

        frames = sample_frames(video_path)
        audio = extract_audio(video_path)

        data = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "frames": frames,
            "audio": audio,
            "label": torch.tensor(label)
        }

        torch.save(data, save_path)


if __name__ == "__main__":

    print("🚀 Preprocessing Train...")
    preprocess_split(
        "data/MELD/train_sent_emo.csv",
        "data/MELD/raw/train",
        "data/MELD/processed/train"
    )

    print("🚀 Preprocessing Dev...")
    preprocess_split(
        "data/MELD/dev_sent_emo.csv",
        "data/MELD/raw/dev",
        "data/MELD/processed/dev"
    )

    print("🚀 Preprocessing Test...")
    preprocess_split(
        "data/MELD/test_sent_emo.csv",
        "data/MELD/raw/test",
        "data/MELD/processed/test"
    )

    print("✅ Preprocessing Complete!")
