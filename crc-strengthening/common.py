"""Shared model, data and metric utilities for the strengthening experiments.

The trained checkpoint (best_model.pt) from the main study is reused as-is; nothing
here retrains it except the optional backbone-ablation script.
"""
import os, re, json, math, random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
TUM_IDX = CLASS_NAMES.index("TUM")
NORM_IDX = CLASS_NAMES.index("NORM")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- model
class CRCNet(nn.Module):
    def __init__(self, backbone="convnext_tiny", num_classes=9, dropout=0.3,
                 hidden_dim=512, tabular_dim=0, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                           num_classes=0, global_pool="")
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self._last_feat = None

    def forward(self, x, return_cam=False):
        fmap = self.backbone.forward_features(x)          # (B,C,h,w)
        if return_cam:
            fmap.retain_grad(); self._last_feat = fmap
        z = fmap.mean(dim=(2, 3))
        z = self.dropout(z)
        return self.classifier(z)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CRCNet(backbone=ck.get("backbone", "convnext_tiny"),
                   num_classes=len(ck.get("class_to_idx", CLASS_NAMES)),
                   dropout=ck.get("dropout", 0.3),
                   hidden_dim=ck.get("hidden_dim", 512),
                   tabular_dim=ck.get("tabular_dim", 0), pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    temp = float(ck.get("temperature", 1.0))
    return model, temp, ck


# --------------------------------------------------------------------------- data
def eval_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(img_size), transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def discover(root):
    root = Path(root)
    classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])
    c2i = {c: i for i, c in enumerate(classes)}
    samples = []
    for c in classes:
        for e in os.scandir(root / c):
            if e.is_file() and Path(e.name).suffix.lower() in IMG_EXTS:
                samples.append((e.path, c2i[c]))
    return samples, c2i


class FolderDS(Dataset):
    def __init__(self, samples, tf):
        self.samples, self.tf = samples, tf

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.tf(Image.open(p).convert("RGB")), y, p


def make_loader(root, batch_size=64, workers=2, img_size=224):
    samples, c2i = discover(root)
    if not samples:
        raise FileNotFoundError(f"No class sub-folders with images under {root}")
    dl = DataLoader(FolderDS(samples, eval_transform(img_size)), batch_size=batch_size,
                    shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())
    return dl, c2i, samples


def slide_id_from_path(path):
    """Extract a slide/patient id from a Kather tile filename.

    Kather tiles look like 'TUM-TCGA-EWDLYHAE.tif'; the token after the class code
    identifies the source slide, enabling slide-clustered bootstrap.
    """
    name = Path(path).stem
    m = re.match(r"^[A-Z]+-([A-Za-z0-9]+)", name)
    if m:
        return m.group(1)
    return name


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def collect_logits(model, loader, device):
    logits, labels, paths = [], [], []
    for x, y, p in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y); paths += list(p)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy(), paths


def softmax_np(logits, T=1.0):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# --------------------------------------------------------------------------- metrics
def bootstrap_ci(values_fn, n=1000, seed=42, groups=None, items=None):
    """Generic bootstrap. If groups given, resample groups (clustered bootstrap)."""
    rng = np.random.default_rng(seed)
    stats = []
    if groups is not None:
        uniq = np.array(sorted(set(groups)))
        g2idx = {g: np.where(np.array(groups) == g)[0] for g in uniq}
        for _ in range(n):
            chosen = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([g2idx[g] for g in chosen])
            stats.append(values_fn(idx))
    else:
        N = items
        for _ in range(n):
            idx = rng.integers(0, N, N)
            stats.append(values_fn(idx))
    stats = np.array(stats)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def set_plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 11, "font.family": "DejaVu Sans", "axes.edgecolor": "#cccccc",
        "axes.grid": True, "grid.color": "#ededed", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False})
    return plt


PALETTE = {"blue": "#3b6ea5", "grey": "#9aa0a6", "red": "#c0603a",
           "green": "#55a868", "purple": "#8172b3"}
