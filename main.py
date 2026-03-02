import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torch.amp import GradScaler
from sklearn.utils.class_weight import compute_class_weight

from config import *
from src.dataset import MELDDataset
from src.model import MultiModalModel
from src.train import train_epoch
from src.evaluate import evaluate


# ==============================
# Dataset Paths (Processed Only)
# ==============================

TRAIN_FOLDER = "data/MELD/processed/train"
DEV_FOLDER = "data/MELD/processed/dev"
TEST_FOLDER = "data/MELD/processed/test"

train_dataset = MELDDataset(TRAIN_FOLDER)
dev_dataset = MELDDataset(DEV_FOLDER)
test_dataset = MELDDataset(TEST_FOLDER)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)


# ==============================
# Model
# ==============================

model = MultiModalModel(
    use_text=True,
    use_audio=True,
    use_vision=True
).to(DEVICE)


# ==============================
# Compute Class Weights
# ==============================

labels = []
for f in os.listdir(TRAIN_FOLDER):
    data = torch.load(os.path.join(TRAIN_FOLDER, f))
    labels.append(data["label"].item())

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

print("Class Weights:", class_weights)


# ==============================
# Optimizer, Scheduler, Loss
# ==============================

optimizer = optim.AdamW(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

scaler = GradScaler("cuda")

best_macro_f1 = 0


# ==============================
# Freeze / Unfreeze Strategy
# ==============================

def freeze_encoders(model):
    for name, param in model.named_parameters():
        if "fusion" not in name and "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


FREEZE_EPOCHS = 2


# ==============================
# Training Loop
# ==============================

print("🚀 Starting Advanced Training...\n")

for epoch in range(EPOCHS):

    if epoch < FREEZE_EPOCHS:
        freeze_encoders(model)
        print("🔒 Encoders Frozen")
    else:
        unfreeze_all(model)
        print("🔓 Encoders Unfrozen")

    loss, acc = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        DEVICE,
        scaler
    )

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {loss:.4f} | Train Acc: {acc:.4f}")

    dev_acc, dev_macro_f1 = evaluate(model, dev_loader, DEVICE, "DEV")

    if dev_macro_f1 > best_macro_f1:
        best_macro_f1 = dev_macro_f1
        torch.save(model.state_dict(), "best_model.pt")
        print("🏆 Best model saved!")

    scheduler.step()


# ==============================
# Final Evaluation
# ==============================

print("📦 Loading Best Model for Final Evaluation...")
model.load_state_dict(torch.load("best_model.pt"))

evaluate(model, test_loader, DEVICE, "TEST")