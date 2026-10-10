"""Cross-dataset external validation on an INDEPENDENT-SOURCE colon set (LC25000).

This is the key 'second dataset / multi-source' experiment. The 9-class model trained
on NCT-CRC-HE-100K is applied, with NO retraining, to colon histology from a different
source. We evaluate the clinically decisive tumour-vs-benign distinction using the
model's P(TUM) as the tumour score:
  - AUC of P(TUM) for adenocarcinoma vs benign (threshold-free, robust to domain shift),
  - accuracy/F1 at the argmax-is-TUM decision and at the Youden-optimal threshold.
Expected folders:  <root>/colon_aca (tumour)  and  <root>/colon_n (benign)
(works equally for Chaoyang 'adenocarcinoma'/'normal' — pass those two folders.)
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

from common import (load_model, get_device, seed_everything, eval_transform, softmax_np,
                    TUM_IDX, FolderDS, IMG_EXTS, set_plot_style, PALETTE)
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, confusion_matrix


def build_samples(root):
    root = Path(root)
    aca = [p for p in (root / "colon_aca").glob("*") if p.suffix.lower() in IMG_EXTS] \
        if (root / "colon_aca").exists() else []
    nrm = [p for p in (root / "colon_n").glob("*") if p.suffix.lower() in IMG_EXTS] \
        if (root / "colon_n").exists() else []
    if not aca or not nrm:   # fallback: two sub-folders alphabetical (aca first by name?)
        subs = sorted([d for d in root.iterdir() if d.is_dir()])
        if len(subs) == 2:
            # assume folder containing 'aca'/'adeno'/'tum'/'cancer' is tumour
            def is_tum(n): return any(k in n.lower() for k in ["aca", "adeno", "tum", "cancer", "malig"])
            t = subs[0] if is_tum(subs[0].name) else subs[1]
            b = subs[1] if t == subs[0] else subs[0]
            aca = [p for p in t.glob("*") if p.suffix.lower() in IMG_EXTS]
            nrm = [p for p in b.glob("*") if p.suffix.lower() in IMG_EXTS]
    samples = [(str(p), 1) for p in aca] + [(str(p), 0) for p in nrm]   # 1=tumour,0=benign
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best_model.pt")
    ap.add_argument("--root", required=True, help="folder with colon_aca/ and colon_n/")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(); Path(a.out).mkdir(exist_ok=True)
    seed_everything(42); device = get_device()
    model, T, _ = load_model(a.ckpt, device)

    samples = build_samples(a.root)
    if not samples:
        raise FileNotFoundError("Could not find colon_aca/colon_n (or two class folders) under " + a.root)
    dl = DataLoader(FolderDS(samples, eval_transform(224)), batch_size=64,
                    shuffle=False, num_workers=2)
    logits, ys = [], []
    with torch.no_grad():
        for x, y, _ in dl:
            logits.append(model(x.to(device)).cpu()); ys.append(y)
    logits = torch.cat(logits).numpy(); ys = torch.cat(ys).numpy()
    probs = softmax_np(logits, T)
    tum_score = probs[:, TUM_IDX]
    pred_argmax = (probs.argmax(1) == TUM_IDX).astype(int)

    auc = roc_auc_score(ys, tum_score)
    fpr, tpr, thr = roc_curve(ys, tum_score)
    j = np.argmax(tpr - fpr); tstar = thr[j]
    pred_opt = (tum_score >= tstar).astype(int)

    res = {
        "dataset": str(a.root), "n_tumour": int((ys == 1).sum()), "n_benign": int((ys == 0).sum()),
        "AUC_tumour_vs_benign": round(float(auc), 4),
        "argmax_is_TUM": {"accuracy": round(accuracy_score(ys, pred_argmax), 4),
                          "f1": round(f1_score(ys, pred_argmax, zero_division=0), 4),
                          "confusion": confusion_matrix(ys, pred_argmax).tolist()},
        "youden_threshold": {"threshold": round(float(tstar), 4),
                             "accuracy": round(accuracy_score(ys, pred_opt), 4),
                             "f1": round(f1_score(ys, pred_opt, zero_division=0), 4),
                             "confusion": confusion_matrix(ys, pred_opt).tolist()},
        "note": ("Zero-shot cross-source transfer; different institution/scanner and "
                 "magnification than NCT-CRC-HE-100K. AUC is the primary metric because it "
                 "is threshold-free and robust to the domain shift.")}
    json.dump(res, open(Path(a.out) / "cross_dataset_lc25000.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    plt = set_plot_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=PALETTE["grey"], ls="--", lw=1)
    ax.plot(fpr, tpr, color=PALETTE["blue"], label=f"AUC = {auc:.3f}")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.legend(frameon=False); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(Path(a.out) / "cross_dataset_roc.pdf"); fig.savefig(Path(a.out) / "cross_dataset_roc.png")
    print("saved cross_dataset_roc.pdf/.png")


if __name__ == "__main__":
    main()
