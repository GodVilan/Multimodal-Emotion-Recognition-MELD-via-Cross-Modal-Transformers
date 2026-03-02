import torch
import torch.nn as nn
from src.encoders import TextEncoder, AudioEncoder, VisionEncoder
from src.fusion import CrossModalFusion
from config import NUM_CLASSES


class MultiModalModel(nn.Module):

    def __init__(self, use_text=True, use_audio=True, use_vision=True):
        super().__init__()

        # Ablation switches
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_vision = use_vision

        # Encoders (only initialize if used)
        if self.use_text:
            self.text_encoder = TextEncoder()

        if self.use_audio:
            self.audio_encoder = AudioEncoder()

        if self.use_vision:
            self.vision_encoder = VisionEncoder()

        self.fusion = CrossModalFusion()

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, frames, audio):

        features = []

        # TEXT
        if self.use_text:
            text_feat = self.text_encoder(input_ids, attention_mask)
            features.append(text_feat)

        # AUDIO
        if self.use_audio:
            audio_feat = self.audio_encoder(audio)
            features.append(audio_feat)

        # VISION
        if self.use_vision:
            vision_feat = self.vision_encoder(frames)
            features.append(vision_feat)

        # Concatenate token sequences along sequence dimension
        combined = torch.cat(features, dim=1)

        # Cross-modal fusion
        fused = self.fusion(combined, combined, combined)

        return self.classifier(fused)
