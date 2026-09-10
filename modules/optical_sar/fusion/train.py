"""Training, validation, and evaluation engine for M3 multimodal model."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from modules.optical_sar.fusion.dataset import CLASS_NAMES
from modules.optical_sar.fusion.model import OpticalSARModel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Structured container for model evaluation metrics."""
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    confusion_matrix: List[List[int]]
    class_names: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision_macro": round(self.precision_macro, 4),
            "precision_weighted": round(self.precision_weighted, 4),
            "recall_macro": round(self.recall_macro, 4),
            "recall_weighted": round(self.recall_weighted, 4),
            "f1_macro": round(self.f1_macro, 4),
            "f1_weighted": round(self.f1_weighted, 4),
            "confusion_matrix": self.confusion_matrix,
            "class_names": self.class_names,
        }


def train_multimodal_model(
    model: OpticalSARModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 15,
    learning_rate: float = 1e-3,
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """Train OpticalSARModel using PyTorch and track validation loss for checkpointing.

    Returns:
        History dictionary with 'train_loss' and 'val_loss' lists.
    """
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    logger.info(f"Initiating training for {num_epochs} epochs on device '{dev}'...")

    for epoch in range(1, num_epochs + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        train_batches = 0

        for opt_x, sar_x, labels in train_loader:
            opt_x = opt_x.to(dev)
            sar_x = sar_x.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            logits = model(opt_x, sar_x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_batches += 1

        avg_train_loss = running_loss / max(train_batches, 1)

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for opt_x, sar_x, labels in val_loader:
                opt_x = opt_x.to(dev)
                sar_x = sar_x.to(dev)
                labels = labels.to(dev)

                logits = model(opt_x, sar_x)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_batches += 1

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / max(val_batches, 1)
        val_acc = correct / max(total, 1)
        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        # 3. Checkpoint Best Weights
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved best model checkpoint to {checkpoint_path}")

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
            f"{' * (Best)' if is_best else ''}"
        )

    # Load best weights into model
    if checkpoint_path is not None and checkpoint_path.is_file():
        model.load_state_dict(torch.load(checkpoint_path, map_location=dev))
        model.has_trained_weights = True

    return history


def evaluate_multimodal_model(
    model: OpticalSARModel,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> EvaluationMetrics:
    """Evaluate trained model on hold-out test set and compute comprehensive metrics."""
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for opt_x, sar_x, labels in test_loader:
            opt_x = opt_x.to(dev)
            sar_x = sar_x.to(dev)
            logits = model(opt_x, sar_x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist()

    return EvaluationMetrics(
        accuracy=acc,
        precision_macro=p_macro,
        precision_weighted=p_weighted,
        recall_macro=r_macro,
        recall_weighted=r_weighted,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        confusion_matrix=cm,
        class_names=CLASS_NAMES,
    )
