import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate(model, loader, device, split_name="DEV"):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["frames"].to(device),
                batch["audio"].to(device)
            )

            predictions = torch.argmax(outputs, dim=1)
            preds.extend(predictions.cpu().numpy())
            labels.extend(batch["label"].numpy())

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    print(f"\n📊 {split_name} RESULTS")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}\n")

    print(classification_report(labels, preds))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{split_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return acc, macro_f1
