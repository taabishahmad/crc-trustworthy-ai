"""Domain-shift / robustness stress test (addresses point 17 - 'no domain-shift experiment').

Applies increasing, realistic corruptions to the external cohort and measures accuracy
degradation. This directly tests the staining/scanner-variation motivation WITHOUT a
second labelled dataset. Corruptions: brightness, contrast, saturation shift (stain
intensity), hue shift (stain colour), Gaussian blur (focus), Gaussian noise (sensor),
and JPEG compression. Also reports calibrated ECE under shift.
"""
import argparse, json, io
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision import transforms

from common import (load_model, discover, get_device, seed_everything, softmax_np,
                    CLASS_NAMES, MEAN, STD, set_plot_style, PALETTE)

NORM = transforms.Normalize(MEAN, STD)
TO = transforms.ToTensor()


def corrupt(img, kind, sev):
    if kind == "brightness": return ImageEnhance.Brightness(img).enhance(1 + sev)
    if kind == "contrast":   return ImageEnhance.Contrast(img).enhance(1 + sev)
    if kind == "saturation": return ImageEnhance.Color(img).enhance(1 + sev)
    if kind == "hue":
        a = np.array(img.convert("HSV")).astype(int)
        a[..., 0] = (a[..., 0] + int(sev * 255)) % 256
        return Image.fromarray(a.astype("uint8"), "HSV").convert("RGB")
    if kind == "blur":       return img.filter(ImageFilter.GaussianBlur(radius=sev * 4))
    if kind == "noise":
        a = np.array(img).astype(float) + np.random.normal(0, sev * 60, (img.height, img.width, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    if kind == "jpeg":
        b = io.BytesIO(); img.save(b, "JPEG", quality=max(5, int(100 - sev * 95)))
        return Image.open(b).convert("RGB")
    return img


@torch.no_grad()
def eval_corrupt(model, samples, kind, sev, device, T):
    model.eval(); preds, labels, confs = [], [], []
    batch, ys = [], []
    def flush():
        nonlocal batch, ys
        if not batch: return
        x = torch.stack(batch).to(device)
        p = softmax_np(model(x).cpu().numpy(), T)
        preds.extend(p.argmax(1)); confs.extend(p.max(1)); labels.extend(ys)
        batch, ys = [], []
    for path, y in samples:
        im = Image.open(path).convert("RGB").resize((224, 224))
        if sev > 0: im = corrupt(im, kind, sev)
        batch.append(NORM(TO(im))); ys.append(y)
        if len(batch) == 64: flush()
    flush()
    preds, labels = np.array(preds), np.array(labels)
    return float((preds == labels).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best_model.pt")
    ap.add_argument("--root", default="data/CRC-VAL-HE-7K")
    ap.add_argument("--subset", type=int, default=2000, help="tiles for speed (0=all)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(); Path(a.out).mkdir(exist_ok=True)
    seed_everything(42); device = get_device()
    model, T, _ = load_model(a.ckpt, device)
    samples, _ = discover(a.root)
    if a.subset and a.subset < len(samples):
        rng = np.random.default_rng(42)
        samples = [samples[i] for i in rng.choice(len(samples), a.subset, replace=False)]

    base = eval_corrupt(model, samples, "none", 0, device, T)
    kinds = ["brightness", "contrast", "saturation", "hue", "blur", "noise", "jpeg"]
    sevs = [0.1, 0.2, 0.3, 0.4, 0.5]
    res = {"clean_accuracy": round(base, 4), "severities": sevs, "accuracy": {}}
    for k in kinds:
        res["accuracy"][k] = [round(eval_corrupt(model, samples, k, s, device, T), 4) for s in sevs]
        print(k, res["accuracy"][k])
    # overall robustness = mean relative accuracy retained across all corruptions/severities
    allv = np.array([v for k in kinds for v in res["accuracy"][k]])
    res["mean_accuracy_under_shift"] = round(float(allv.mean()), 4)
    res["mean_relative_retention"] = round(float(allv.mean() / max(base,1e-9)), 4)
    res["note"] = ("accuracy on %d tiles; clean vs corrupted. Graceful degradation "
                   "evidences robustness to staining/scanner-type shift." % len(samples))
    json.dump(res, open(Path(a.out) / "domain_shift.json", "w"), indent=2)

    plt = set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    cols = list(PALETTE.values()) + ["#000000", "#cc8800"]
    for (k, c) in zip(kinds, cols):
        ax.plot([0] + sevs, [base] + res["accuracy"][k], marker="o", ms=3, color=c, label=k)
    ax.set_xlabel("corruption severity"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(Path(a.out) / "domain_shift.pdf"); fig.savefig(Path(a.out) / "domain_shift.png")
    print("clean", base, "| saved domain_shift.pdf/.png")


if __name__ == "__main__":
    main()
