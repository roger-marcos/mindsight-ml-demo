from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

from config import ROOT_DIR, TARGET_CLASSES, set_seed
from data import get_dataloaders
from train import build_model, get_device


def load_best_model(num_classes: int, device: torch.device):
    model_path = Path(ROOT_DIR) / "models" / "resnet18_cifar3_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Make sure you have run train.py first."
        )

    model = build_model(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_on_test():
    set_seed()
    device = get_device()

    # We only need the test loader here
    _, _, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    model = load_best_model(num_classes=num_classes, device=device)

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    # Compute metrics
    acc = accuracy_score(all_targets, all_preds)
    print(f"\nTest accuracy: {acc * 100:.2f}%\n")

    print("Classification report:")
    print(
        classification_report(
            all_targets,
            all_preds,
            target_names=class_names,
            digits=3,
        )
    )

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion matrix:")
    print(cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")

    # Add values on cells
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(im, ax=ax)

    out_path = Path(ROOT_DIR) / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved confusion matrix to {out_path}")


if __name__ == "__main__":
    evaluate_on_test()
