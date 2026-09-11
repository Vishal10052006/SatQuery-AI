"""Train/evaluate the SatQuery Siamese U-Net change detector."""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.change.siamese_unet import SiameseUNet
from training.dataset import BitemporalChangeDataset


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (den + 1.0)).mean()


def confusion(logits, target):
    pred = torch.sigmoid(logits) >= 0.5
    truth = target >= 0.5
    return (
        int((pred & truth).sum().item()),
        int((pred & ~truth).sum().item()),
        int((~pred & truth).sum().item()),
    )


def scores(tp, fp, fn):
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    return f1, iou


def run_epoch(model, loader, optimizer, device, scaler, train, amp, log_every):
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    tp = fp = fn = 0
    start = time.time()

    for step, batch in enumerate(loader, 1):
        before = batch["before"].to(device, non_blocking=True)
        after = batch["after"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(before, after)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        btp, bfp, bfn = confusion(logits.detach(), target)
        tp += btp
        fp += bfp
        fn += bfn

        bs = before.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        if train and (step == 1 or step % log_every == 0 or step == len(loader)):
            elapsed = time.time() - start
            pct = 100.0 * step / len(loader)
            print(
                f"\r  train {step:4d}/{len(loader)} ({pct:5.1f}%) "
                f"loss={loss.item():.4f} time={elapsed/60:.1f}m",
                end="",
                flush=True,
            )

    if train:
        print()

    f1, iou = scores(tp, fp, fn)
    return total_loss / max(total_samples, 1), f1, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--checkpoint", default="checkpoints/rcd/building_change_siamese_unet.pth")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(f"amp={amp}")

    train_ds = BitemporalChangeDataset(args.data_root, "train", args.patch_size, True)
    val_ds = BitemporalChangeDataset(args.data_root, "val", args.patch_size, False)
    print(f"train_samples={len(train_ds)}")
    print(f"val_samples={len(val_ds)}")

    tl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    vl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model = SiameseUNet(in_channels=3, base=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        print(f"\nEpoch {epoch}/{args.epochs} lr={optimizer.param_groups[0]['lr']:.2e}")

        train_loss, train_f1, train_iou = run_epoch(
            model, tl, optimizer, device, scaler, True, amp, args.log_every
        )
        val_loss, val_f1, val_iou = run_epoch(
            model, vl, optimizer, device, scaler, False, amp, args.log_every
        )
        scheduler.step()

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"train_f1={train_f1:.4f} train_iou={train_iou:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} "
            f"val_iou={val_iou:.4f} time={(time.time()-started)/60:.1f}m"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {"in_channels": 3, "base": 32},
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "val_iou": val_iou,
                    "train_f1": train_f1,
                    "train_iou": train_iou,
                },
                checkpoint,
            )
            print(f"  BEST CHECKPOINT SAVED -> {checkpoint} (F1={val_f1:.4f})")

    print(f"\nbest_checkpoint={checkpoint}")
    print(f"best_val_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
