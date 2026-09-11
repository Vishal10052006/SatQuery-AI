"""Multimodal Optical + SAR model wrapper.

Predictions are emitted only when trained weights are successfully loaded.
An untrained network is an architecture baseline, not a scientific predictor.
"""
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.optical_sar.fusion.feature_fusion import ConvBlock, FeatureFusionNetwork

logger = logging.getLogger(__name__)

@dataclass
class ModelInferenceResult:
    predicted_class: Optional[int]
    probabilities: Optional[List[float]]
    logits: Optional[List[float]]
    model_confidence: Optional[float]
    fusion_type: str
    status: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "probabilities": [round(p, 4) for p in self.probabilities] if self.probabilities else None,
            "logits": [round(x, 4) for x in self.logits] if self.logits else None,
            "model_confidence": round(self.model_confidence, 4) if self.model_confidence is not None else None,
            "fusion_type": self.fusion_type,
            "status": self.status,
            "notes": self.notes,
        }

class EarlyFusionNetwork(nn.Module):
    def __init__(self, in_channels: int = 6, feature_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, pool=True),
            ConvBlock(32, 64, kernel_size=3, pool=True),
            ConvBlock(64, feature_dim, kernel_size=3, pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder(x).flatten(1))

class OpticalSARModel(nn.Module):
    """Multimodal model supporting feature and early fusion."""
    def __init__(self, fusion_type: str = "feature", optical_channels: int = 4,
                 sar_channels: int = 2, feature_dim: int = 128, num_classes: int = 3,
                 weights_path: Optional[str] = None, device: Optional[str] = None):
        super().__init__()
        self.fusion_type = fusion_type
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.has_trained_weights = False
        self.weights_error: Optional[str] = None

        if fusion_type == "feature":
            self.network = FeatureFusionNetwork(optical_channels, sar_channels, feature_dim, num_classes)
        elif fusion_type == "early":
            self.network = EarlyFusionNetwork(optical_channels + sar_channels, feature_dim, num_classes)
        else:
            raise ValueError(f"Unknown fusion_type '{fusion_type}'. Choose 'feature' or 'early'.")

        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(self.device)
        if weights_path is not None:
            self._load_weights(weights_path)

    def _load_weights(self, path: str) -> None:
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict)
            self.has_trained_weights = True
            logger.info("Loaded trained model weights from %s", path)
        except Exception as exc:
            self.has_trained_weights = False
            self.weights_error = str(exc)
            logger.warning("Could not load weights from %s: %s", path, exc)

    def forward(self, optical: Union[torch.Tensor, np.ndarray], sar: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(optical, np.ndarray):
            optical = torch.from_numpy(optical).float()
        if isinstance(sar, np.ndarray):
            sar = torch.from_numpy(sar).float()
        if optical.dim() == 3:
            optical = optical.unsqueeze(0)
        if sar.dim() == 3:
            sar = sar.unsqueeze(0)
        if optical.dim() != 4 or sar.dim() != 4:
            raise ValueError("Optical and SAR inputs must be 3-D or 4-D tensors")
        if optical.shape[0] != sar.shape[0] or optical.shape[-2:] != sar.shape[-2:]:
            raise ValueError("Optical and SAR inputs must have matching batch and spatial dimensions")
        if optical.shape[1] != self.optical_channels or sar.shape[1] != self.sar_channels:
            raise ValueError(
                f"Channel mismatch: expected optical={self.optical_channels}, sar={self.sar_channels}; "
                f"received optical={optical.shape[1]}, sar={sar.shape[1]}"
            )
        optical = torch.nan_to_num(optical.to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
        sar = torch.nan_to_num(sar.to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
        return self.network(optical, sar) if self.fusion_type == "feature" else self.network(torch.cat([optical, sar], dim=1))

    @torch.no_grad()
    def predict(self, optical: Union[torch.Tensor, np.ndarray], sar: Union[torch.Tensor, np.ndarray]) -> ModelInferenceResult:
        """Run inference only with successfully loaded trained weights."""
        if not self.has_trained_weights:
            note = "Trained multimodal weights are unavailable; no scientific class prediction was emitted."
            if self.weights_error:
                note += f" Weight loading failed: {self.weights_error}"
            return ModelInferenceResult(None, None, None, None, self.fusion_type, "not_available", note)

        self.eval()
        logits = self.forward(optical, sar)
        probs = F.softmax(logits, dim=-1)
        probs_np = probs.cpu().numpy()[0]
        logits_np = logits.cpu().numpy()[0]
        return ModelInferenceResult(
            predicted_class=int(np.argmax(probs_np)),
            probabilities=probs_np.tolist(),
            logits=logits_np.tolist(),
            model_confidence=float(np.max(probs_np)),
            fusion_type=self.fusion_type,
            status="success",
            notes="Inference executed with trained model weights; probabilities are model outputs and are not assumed calibrated.",
        )
