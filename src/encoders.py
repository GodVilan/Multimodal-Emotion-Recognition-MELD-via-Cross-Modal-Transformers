import torch.nn as nn
from transformers import BertModel, Wav2Vec2Model, ViTModel

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state  # (B, T, 768)

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    def forward(self, audio):
        output = self.model(audio)
        return output.last_hidden_state  # (B, T, 768)

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")

    def forward(self, frames):
        B, F, C, H, W = frames.shape
        frames = frames.view(B * F, C, H, W)
        output = self.model(frames).last_hidden_state[:, 0, :]
        output = output.view(B, F, -1)
        return output  # (B, F, 768)
