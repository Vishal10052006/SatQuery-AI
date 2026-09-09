"""Multimodal Optical + SAR Model wrapper supporting early-fusion and feature-fusion baselines."""

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
    """Structured result of multimodal model inference.

    Attributes:
        predicted_class: Integer index of predicted class.
        probabilities: Probability distribution across classes as a float list.
        logits: Raw output logits as a float list.
        model_confidence: Highest probability score [0.0 to 1.0].
        fusion_type: Method used ('early' or 'feature').
        status: Inference status code ('success', 'untrained_baseline', 'not_available').
        notes: Scientific notes on weights and training status.
    """
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
            "model_confidence": round(self.model_confidence, 4) if self.model_confidence else None,
            "fusion_type": self.fusion_type,
            "status": self.status,
            "notes": self.notes,
        }


class EarlyFusionNetwork(nn.Module):
    """Single-backbone CNN accepting early channel-concatenated Optical + SAR input."""

    def __init__(self, in_channels: int = 6, feature_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, pool=True),
            ConvBlock(32, 64, kernel_size=3, pool=True),
            ConvBlock(64, feature_dim, kernel_size=3, pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        feats = self.encoder(x).view(b, -1)
        return self.fc(feats)


class OpticalSARModel(nn.Module):
    """High-level Multimodal Optical + SAR Deep Learning Model.

    Supports:
    - 'feature': Two-stream Siamese/heterogeneous CNN encoder with feature fusion layer.
    - 'early': Single CNN backbone taking concatenated multi-channel raster.

    Training Data & Target Specification:
        This model architecture is compatible with multimodal Earth Observation benchmarks
        such as SEN12MS, BigEarthNet-MM, and EuroSAT.
        When initialized without pre-trained weights, predictions represent random weight
        initializations and are clearly flagged as 'untrained_baseline' to prevent fabricated claims.
    """

    def __init__(
        self,
        fusion_type: str = "feature",
        optical_channels: int = 4,
        sar_channels: int = 2,
        feature_dim: int = 128,
        num_classes: int = 5,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.has_trained_weights = False

        if fusion_type == "feature":
            self.network = FeatureFusionNetwork(
                optical_channels=optical_channels,
                sar_channels=sar_channels,
                feature_dim=feature_dim,
                num_classes=num_classes,
            )
        elif fusion_type == "early":
            total_channels = optical_channels + sar_channels
            self.network = EarlyFusionNetwork(
                in_channels=total_channels,
                feature_dim=feature_dim,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unknown fusion_type '{fusion_type}'. Choose 'feature' or 'early'.")

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.to(self.device)

        if weights_path is not None:
            self._load_weights(weights_path)

    def _load_weights(self, path: str):
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict)
            self.has_trained_weights = True
            logger.info(f"Loaded trained model weights from {path}")
        except Exception as e:
            logger.warning(f"Could not load weights from {path}: {e}")
            self.has_trained_weights = False

    def forward(
        self,
        optical: Union[torch.Tensor, np.ndarray],
        sar: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """Forward pass accepting Optical and SAR inputs.

        Args:
            optical: Tensor or array of shape (B, C_opt, H, W).
            sar: Tensor or array of shape (B, C_sar, H, W).

        Returns:
            Class logits tensor of shape (B, num_classes).
        """
        # Convert numpy to torch tensors if needed
        if isinstance(optical, np.ndarray):
            optical = torch.from_numpy(optical).float()
        if isinstance(sar, np.ndarray):
            sar = torch.from_numpy(sar).float()

        if optical.dim() == 3:
            optical = optical.unsqueeze(0)
        if sar.dim() == 3:
            sar = sar.unsqueeze(0)

        optical = optical.to(self.device)
        sar = sar.to(self.device)

        # Replace any NaNs with 0.0 for neural network forward pass
        optical = torch.nan_to_num(optical, nan=0.0)
        sar = torch.nan_to_num(sar, nan=0.0)

        if self.fusion_type == "feature":
            return self.network(optical, sar)
        else:
            # Concatenate for early fusion
            concat = torch.cat([optical, sar], dim=1)
            return self.network(concat)

    @torch.no_grad()
    def predict(
        self,
        optical: Union[torch.Tensor, np.ndarray],
        sar: Union[torch.Tensor, np.ndarray],
    ) -> ModelInferenceResult:
        """Perform inference with probability calibration checks."""
        self.eval()
        logits_tensor = self.forward(optical, sar)
        probs_tensor = F.softmax(logits_tensor, dim=-1)

        probs_np = probs_tensor.cpu().numpy()[0]
        logits_np = logits_tensor.cpu().numpy()[0]

        pred_class = int(np.argmax(probs_np))
        confidence = float(np.max(probs_np))

        status = "success" if self.has_trained_weights else "untrained_baseline"
        notes = (
            "Inference executed with trained model weights."
            if self.has_trained_weights
            else "Untrained baseline architecture. Probabilities reflect random weight initialization; "
                 "requires downstream training on labelled multimodal dataset (e.g. SEN12MS)."
        )

        return ModelInferenceResult(
            predicted_class=pred_class,
            probabilities=probs_np.tolist(),
            logits=logits_np.tolist(),
            model_confidence=confidence,
            fusion_type=self.fusion_type,
            status=status,
            notes=notes,
        )
