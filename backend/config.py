import os
from pathlib import Path
import torch

# Root directory of this backend folder
ROOT_DIR = Path(__file__).resolve().parent

# Where to download/store CIFAR-10
DATA_DIR = ROOT_DIR / "data"

# Training configuration
BATCH_SIZE = 64
NUM_WORKERS = 2  # you can set 0 if you get issues on macOS
VAL_SPLIT = 0.2  # 20% of training subset for validation
RANDOM_SEED = 42

# Target classes from CIFAR-10
TARGET_CLASSES = ["airplane", "automobile", "ship"]

# For reproducibility
def set_seed(seed: int = RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (optional, might slow a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
