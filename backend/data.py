from typing import Tuple, List
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, VAL_SPLIT, TARGET_CLASSES, set_seed


def get_transforms():
    """Return data augmentations for train and simple transforms for test/val."""
    # CIFAR-10 images are 32x32 RGB
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


class RemappedSubset(Dataset):
    """
    Wrap a dataset and:
      - keep only selected indices
      - remap original CIFAR-10 labels to [0, 1, 2, ...]
    """

    def __init__(self, base_dataset, indices, orig_to_new_label):
        self.base_dataset = base_dataset
        self.indices = indices
        self.orig_to_new_label = orig_to_new_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base_dataset[self.indices[idx]]
        new_y = self.orig_to_new_label[int(y)]
        return x, new_y


def _build_filtered_dataset(dataset, target_classes: List[str]) -> RemappedSubset:
    """
    Given a CIFAR-10 dataset, return a dataset that:
      - only contains target_classes
      - remaps their original labels to [0..len(target_classes)-1]
    """
    all_classes = dataset.classes  # e.g. ['airplane', 'automobile', 'bird', ...]
    class_name_to_orig = {name: idx for idx, name in enumerate(all_classes)}

    # original label indices for our selected classes, e.g. [0, 1, 8]
    selected_orig_labels = [class_name_to_orig[c] for c in target_classes]

    # mapping original label -> new label (0,1,2)
    orig_to_new_label = {
        orig_label: new_idx
        for new_idx, orig_label in enumerate(selected_orig_labels)
    }

    # keep only samples whose label is in selected_orig_labels
    indices = [i for i, y in enumerate(dataset.targets) if y in selected_orig_labels]

    return RemappedSubset(dataset, indices, orig_to_new_label)


def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Download CIFAR-10 if needed, filter to 3 classes, and return:
    train_loader, val_loader, test_loader, class_names
    where labels are remapped to 0..2
    """
    set_seed()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_transform, test_transform = get_transforms()

    # Full CIFAR-10 train and test
    full_train = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=train_transform,
    )

    full_test = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=test_transform,
    )

    # Filter + remap labels
    train_filtered = _build_filtered_dataset(full_train, TARGET_CLASSES)
    test_filtered = _build_filtered_dataset(full_test, TARGET_CLASSES)

    class_names = TARGET_CLASSES  # order defines label 0,1,2

    # Split train_filtered into train + val
    n_total = len(train_filtered)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        train_filtered,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_filtered,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    print("Building dataloaders...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print("Classes:", class_names)

    batch = next(iter(train_loader))
    images, labels = batch
    print("Train batch shape:", images.shape)   # [B, 3, 32, 32]
    print("Train batch labels shape:", labels.shape)
    print("Unique labels in this batch:", labels.unique())  # should be 0,1,2
    print("Done.")
