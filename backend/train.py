import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import accuracy_score

from data import get_dataloaders
from config import set_seed, ROOT_DIR


def get_device() -> torch.device:
    """Choose the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
    """
    Build a ResNet18 model with pretrained weights and replace the final layer
    to predict `num_classes` classes.
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)

    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)

    return epoch_loss, epoch_acc


def main():
    set_seed()
    device = get_device()

    print("Building dataloaders...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print(f"Classes: {class_names} (num_classes={num_classes})")

    print("Building model...")
    model = build_model(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = 10

    best_val_acc = 0.0
    best_model_path = Path(ROOT_DIR) / "models" / "resnet18_cifar3_best.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%\n"
            f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%"
        )

        # Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  👉 New best model saved to {best_model_path} (val_acc={val_acc*100:.2f}%)")

    # Load best model before testing
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Final test accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
