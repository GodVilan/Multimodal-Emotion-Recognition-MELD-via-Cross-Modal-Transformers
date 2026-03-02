import torch.nn as nn
from config import EMBED_DIM


class CrossModalFusion(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

    def forward(self, combined, *args):
        fused = self.transformer(combined)
        return fused.mean(dim=1)
