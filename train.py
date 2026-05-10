import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset" / "images"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_WORKERS = 0
SEED = 42


@dataclass
class TrainConfig:
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    lr: float = 1e-4
    fine_tune_lr: float = 1e-5
    weight_decay: float = 1e-4
    t_max: int = 20
    seed: int = SEED


class PetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: Dict[str, int], transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[row["class_name"]]
        return image, label


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_class_name(image_path: Path) -> str:
    stem = image_path.stem
    match = re.match(r"^(.*)_\d+$", stem)
    if not match:
        raise ValueError(f"Invalid file name format: {image_path.name}")
    return match.group(1)


def build_dataframe(dataset_dir: Path) -> pd.DataFrame:
    image_paths = sorted(dataset_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg files found in {dataset_dir}")

    records = []
    for image_path in image_paths:
        class_name = parse_class_name(image_path)
        records.append({"image_path": str(image_path), "class_name": class_name})

    df = pd.DataFrame(records)
    print(f"Total images: {len(df)}")
    print(f"Total classes found: {df['class_name'].nunique()}")
    return df


def stratified_split(df: pd.DataFrame, seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["class_name"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["class_name"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def build_model(num_classes: int):
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    for param in model.parameters():
        param.requires_grad = False

    classifier_modules = []
    if hasattr(model, "classifier"):
        classifier_modules.append(model.classifier)
    if hasattr(model, "fc"):
        classifier_modules.append(model.fc)
    if hasattr(model, "head"):
        classifier_modules.append(model.head)

    for module in classifier_modules:
        for param in module.parameters():
            param.requires_grad = True

    return model


def unfreeze_last_blocks(model):
    for name, param in model.named_parameters():
        if any(key in name.lower() for key in ["blocks.6", "blocks.7", "conv_head", "classifier", "fc", "head"]):
            param.requires_grad = True


def get_trainable_params(model, lr: float, fine_tune_lr: float, fine_tune: bool):
    if not fine_tune:
        return [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": lr},
        ]

    classifier_params = []
    fine_tune_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name.lower() for key in ["classifier", "fc", "head"]):
            classifier_params.append(param)
        else:
            fine_tune_params.append(param)

    param_groups = []
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": lr})
    if fine_tune_params:
        param_groups.append({"params": fine_tune_params, "lr": fine_tune_lr})
    return param_groups


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    top5_correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

        top5 = torch.topk(outputs, k=min(5, outputs.size(1)), dim=1).indices
        top5_correct += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total, top5_correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return running_loss / total


def plot_history(history: dict, output_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy History")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_best_model(model, class_names: List[str], num_classes: int, output_path: Path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": num_classes,
            "model_name": MODEL_NAME,
        },
        output_path,
    )


def main():
    set_seed()
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = build_dataframe(DATASET_DIR)
    train_df, val_df, test_df = stratified_split(df, seed=config.seed)

    class_names = sorted(df["class_name"].unique().tolist())
    num_classes = len(class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_transform, eval_transform = get_transforms()
    train_dataset = PetDataset(train_df, class_to_idx, transform=train_transform)
    val_dataset = PetDataset(val_df, class_to_idx, transform=eval_transform)
    test_dataset = PetDataset(test_df, class_to_idx, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.t_max)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "best_model.pth"

    for epoch in range(config.epochs):
        if epoch == 5:
            unfreeze_last_blocks(model)
            param_groups = get_trainable_params(
                model,
                lr=config.lr,
                fine_tune_lr=config.fine_tune_lr,
                fine_tune=True,
            )
            optimizer = AdamW(param_groups, weight_decay=config.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=config.t_max)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, device)
        train_eval_loss, train_acc, train_top5 = evaluate(model, train_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f} | Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best_model(model, class_names, num_classes, best_model_path)

    plot_history(history, MODELS_DIR / "training_history.png")

    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    test_loss, test_acc, test_top5 = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Top-5 Accuracy: {test_top5 * 100:.2f}%")

    print(f"Best model saved to: {best_model_path}")
    print(f"Class names saved to: {MODELS_DIR / 'class_names.json'}")
    print(f"Training history saved to: {MODELS_DIR / 'training_history.png'}")


if __name__ == "__main__":
    main()
