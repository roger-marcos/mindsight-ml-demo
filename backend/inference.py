from typing import Dict, Tuple
from pathlib import Path
from io import BytesIO
import os
import gdown

import torch
import torch.nn.functional as F
from PIL import Image

from config import ROOT_DIR, TARGET_CLASSES
from data import get_transforms
from train import build_model, get_device


# Cache global objects so they are loaded only once
_device = None
_model = None
_test_transform = None
_class_names = TARGET_CLASSES

MODEL_PATH = Path(ROOT_DIR) / "models" / "resnet18_cifar3_best.pt"


def get_or_load_model():
    global _device, _model, _test_transform

    if _device is None:
        _device = get_device()

    if _test_transform is None:
        _, _test_transform = get_transforms()

    if _model is None:
        # Check if model file exists
        if not MODEL_PATH.exists():
            print(f"Model not found at {MODEL_PATH}. Downloading from Google Drive...")
            # Google Drive direct download link
            gdown.download("https://drive.google.com/uc?export=download&id=1IlVj9vJLsXV6teYzhEId58-tlxp4q3zw", str(MODEL_PATH), quiet=False)
            print("Model downloaded successfully.")

        # Load the model architecture
        num_classes = len(_class_names)
        model = build_model(num_classes=num_classes)

        # Load the state dictionary (model weights)
        state_dict = torch.load(MODEL_PATH, map_location=_device)
        model.load_state_dict(state_dict)
        model.to(_device)
        model.eval()
        _model = model

    return _model, _device, _test_transform


def read_image_from_bytes(data: bytes) -> Image.Image:
    """Load an RGB image from raw bytes."""
    img = Image.open(BytesIO(data)).convert("RGB")
    return img


def predict_image_bytes(data: bytes) -> Tuple[str, Dict[str, float]]:
    """
    Given raw image bytes, return:
      - predicted class name
      - dict of class_name -> probability
    """
    model, device, test_transform = get_or_load_model()

    img = read_image_from_bytes(data)
    tensor = test_transform(img).unsqueeze(0).to(device)  # [1, 3, 32, 32]

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]  # [num_classes]

    probs_list = probs.cpu().tolist()
    pred_idx = int(torch.argmax(probs).item())
    pred_class = _class_names[pred_idx]

    probs_dict = {
        cls_name: float(p)
        for cls_name, p in zip(_class_names, probs_list)
    }

    return pred_class, probs_dict
