# M2 RCD training

SatQuery-AI includes a PyTorch Siamese U-Net training and inference path for building-change detection.

## Dataset
Use LEVIR-CD or another paired binary change dataset with `train/A`, `train/B`, `train/label`, `val/A`, `val/B`, `val/label`, `test/A`, `test/B`, `test/label`. The same filename must exist in A, B and label.

The model reports building-related change; it does not guarantee new-construction versus demolition direction.

## Train
`python -m training.train_rcd --data-root /path/to/LEVIR-CD --epochs 50 --batch-size 8 --patch-size 256 --checkpoint checkpoints/rcd/building_change_siamese_unet.pth`

The best validation-F1 checkpoint is saved at that path. CUDA is used automatically when available.

## Runtime
Once the checkpoint exists, M2 automatically loads it for construction/building queries. Override with `SATQUERY_RCD_CHECKPOINT=/absolute/path/to/model.pth`.

If the semantic checkpoint is missing, the UI explicitly labels the result as a fallback rather than pretending it is a trained semantic prediction.

## Verification
`pytest -q tests/change/test_rcd_neural.py`
