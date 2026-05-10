"""Pet classifier using a PyTorch EfficientNet-B0 model trained with timm.

This module loads the trained model once at import time and exposes
`predict_pet(image_path)` for inference.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import timm

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
BREEDS_JSON_PATH = BASE_DIR / "breeds_vi.json"

IMAGE_SIZE = 224
TOP_K = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

MODEL = None
CLASS_NAMES: list[str] = []
BREEDS_DATA: dict[str, dict[str, str]] = {"dogs": {}, "cats": {}}


def _normalize_key(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def _load_class_names() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay file class_names: {CLASS_NAMES_PATH}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("class_names.json khong dung dinh dang list[str].")

    return data


def _load_breeds_data() -> dict[str, dict[str, str]]:
    if not BREEDS_JSON_PATH.exists():
        return {"dogs": {}, "cats": {}}

    with open(BREEDS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    dogs = data.get("dogs", {}) if isinstance(data, dict) else {}
    cats = data.get("cats", {}) if isinstance(data, dict) else {}

    if not isinstance(dogs, dict):
        dogs = {}
    if not isinstance(cats, dict):
        cats = {}

    return {"dogs": dogs, "cats": cats}


def _build_name_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for group in (BREEDS_DATA.get("dogs", {}), BREEDS_DATA.get("cats", {})):
        for key, name_vi in group.items():
            lookup[_normalize_key(key)] = name_vi
    return lookup


BREEDS_DATA = _load_breeds_data()
NAME_LOOKUP = _build_name_lookup()


def _build_model() -> torch.nn.Module:
    global CLASS_NAMES

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError("File best_model.pth khong co dinh dang checkpoint hop le.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise KeyError("Checkpoint khong chua khoa model_state_dict.")

    class_names = checkpoint.get("class_names", CLASS_NAMES)
    if not isinstance(class_names, list) or not all(isinstance(x, str) for x in class_names):
        raise ValueError("Checkpoint class_names khong hop le.")

    num_classes = len(class_names)
    if num_classes <= 0:
        raise ValueError("num_classes khong hop le.")

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=num_classes,
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    CLASS_NAMES = class_names
    return model


def _get_vi_name(name_en: str) -> str:
    normalized = _normalize_key(name_en)
    return NAME_LOOKUP.get(normalized, name_en)


def _preprocess_image(image_path: str) -> torch.Tensor:
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = _PREPROCESS(img)
        return tensor
    except Exception as e:
        raise ValueError(f"Khong doc duoc anh: {e}") from e


def _ensure_model_loaded() -> None:
    global MODEL, CLASS_NAMES

    if MODEL is not None:
        return

    CLASS_NAMES = _load_class_names()
    MODEL = _build_model()
    print(f"Loaded PyTorch model on {DEVICE} with {len(CLASS_NAMES)} classes.")


try:
    _ensure_model_loaded()
except Exception as exc:
    print(f"[classifier] Model loading failed: {exc}")


def predict_pet(image_path: str) -> dict[str, Any]:
    """Predict pet breed from an image file.

    Returns a dict with top-1 result and top-3 alternatives.
    """
    try:
        _ensure_model_loaded()
        if MODEL is None:
            return {"error": "true", "message": "Khong load duoc model."}

        if not CLASS_NAMES:
            return {"error": "true", "message": "Khong co class_names hop le."}

        img_tensor = _preprocess_image(image_path).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = MODEL(img_tensor)
            if outputs.ndim != 2:
                raise ValueError(f"Output model khong hop le: shape={tuple(outputs.shape)}")

            row = outputs[0]
            row_sum = float(row.sum().item())
            row_min = float(row.min().item())
            row_max = float(row.max().item())

            looks_like_probs = (
                row_min >= 0.0
                and row_max <= 1.0 + 1e-6
                and abs(row_sum - 1.0) <= 1e-3
            )

            if looks_like_probs:
                print(f"DEBUG confidence raw probabilities: sum={row_sum:.6f}, min={row_min:.6f}, max={row_max:.6f}")
                probs = row
            else:
                probs = F.softmax(outputs, dim=1)[0]
                print(
                    f"DEBUG confidence raw logits: sum={row_sum:.6f}, min={row_min:.6f}, max={row_max:.6f}"
                )
                print(
                    f"DEBUG confidence after softmax: sum={float(probs.sum().item()):.6f}, "
                    f"min={float(probs.min().item()):.6f}, max={float(probs.max().item()):.6f}"
                )

            top_probs, top_indices = torch.topk(probs, k=min(TOP_K, probs.shape[0]))

        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            if idx < 0 or idx >= len(CLASS_NAMES):
                continue
            name_en = CLASS_NAMES[idx]
            name_vi = _get_vi_name(name_en)
            confidence_pct = max(0.0, min(100.0, float(prob) * 100.0))
            predictions.append(
                {
                    "name_vi": name_vi,
                    "name_en": name_en,
                    "confidence": round(confidence_pct, 1),
                }
            )

        if not predictions:
            return {"error": "true", "message": "Khong the tao du doan hop le."}

        top1 = predictions[0]
        return {
            "top1_name_vi": top1["name_vi"],
            "top1_name_en": top1["name_en"],
            "top1_confidence": top1["confidence"],
            "alternatives": predictions[1:],
        }

    except FileNotFoundError as e:
        return {"error": "true", "message": str(e)}
    except ValueError as e:
        return {"error": "true", "message": str(e)}
    except Exception as e:
        return {"error": "true", "message": f"Loi nhan dien: {e}"}
