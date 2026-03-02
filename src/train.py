import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.amp import autocast


def train_epoch(model, loader, optimizer, criterion, device, scaler):

    model.train()
    total_loss = 0
    preds_all = []
    labels_all = []

    for batch in tqdm(loader):

        optimizer.zero_grad()

        with autocast("cuda"):

            outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["frames"].to(device),
                batch["audio"].to(device)
            )

            loss = criterion(outputs, batch["label"].to(device))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        preds_all.extend(preds.detach().cpu().numpy())
        labels_all.extend(batch["label"].numpy())

    acc = accuracy_score(labels_all, preds_all)

    return total_loss / len(loader), acc