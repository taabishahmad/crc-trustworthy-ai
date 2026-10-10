"""OPTIONAL (heavy): backbone ablation for the comparison table (Major Comment 5).

Trains additional backbones with the SAME pipeline so the comparison is controlled
(same data split, augmentation, optimiser, schedule, label smoothing). Requires the
full NCT-CRC-HE-100K training set and a GPU (~1-2 h per backbone on a T4).

    python exp6_backbone_ablation_train.py --backbone efficientnet_b0 \
        --root_100k data/NCT-CRC-HE-100K --root_7k data/CRC-VAL-HE-7K

Run once per backbone in {efficientnet_b0, resnet50, convnext_tiny}. Each run writes
results/ablation_<backbone>.json with external accuracy/macro-F1. Also runs a quick
'no-label-smoothing' variant via --label_smoothing 0.0 for that ablation row.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from common import (CRCNet, discover, FolderDS, eval_transform, get_device, seed_everything,
                    MEAN, STD, CLASS_NAMES)


def train_tf(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--root_100k", required=True)
    ap.add_argument("--root_7k", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--subset", type=float, default=1.0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(); Path(a.out).mkdir(exist_ok=True)
    seed_everything(42); device = get_device()

    samples, c2i = discover(a.root_100k)
    if a.subset < 1.0:
        y = [l for _, l in samples]
        samples, _ = train_test_split(samples, train_size=a.subset, stratify=y, random_state=42)
    y = [l for _, l in samples]
    tr, va = train_test_split(samples, test_size=0.15, stratify=y, random_state=42)
    ext, _ = discover(a.root_7k)

    dl_tr = DataLoader(FolderDS(tr, train_tf()), batch_size=a.bs, shuffle=True,
                       num_workers=2, drop_last=True, pin_memory=True)
    dl_va = DataLoader(FolderDS(va, eval_transform()), batch_size=a.bs, num_workers=2)
    dl_ext = DataLoader(FolderDS(ext, eval_transform()), batch_size=a.bs, num_workers=2)

    model = CRCNet(backbone=a.backbone, num_classes=len(c2i), pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(dl_tr); warm = 2 * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
             else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    @torch.no_grad()
    def evaluate(dl):
        model.eval(); P, Y = [], []
        for x, yy, _ in dl:
            P.append(model(x.to(device)).argmax(1).cpu()); Y.append(yy)
        P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
        return accuracy_score(Y, P), f1_score(Y, P, average="macro",
                 labels=list(range(len(c2i))), zero_division=0)

    best = 0.0
    for ep in range(1, a.epochs + 1):
        model.train()
        for x, yy, _ in dl_tr:
            x, yy = x.to(device), yy.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), yy)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        va_acc, _ = evaluate(dl_va)
        best = max(best, va_acc)
        print(f"epoch {ep:02d} val_acc {va_acc:.4f}")

    acc, mf1 = evaluate(dl_ext)
    tag = f"{a.backbone}" + ("" if a.label_smoothing > 0 else "_nosmooth")
    res = {"backbone": a.backbone, "label_smoothing": a.label_smoothing,
           "external_accuracy": round(acc, 4), "external_macro_f1": round(mf1, 4),
           "best_val_acc": round(best, 4), "subset": a.subset}
    json.dump(res, open(Path(a.out) / f"ablation_{tag}.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
