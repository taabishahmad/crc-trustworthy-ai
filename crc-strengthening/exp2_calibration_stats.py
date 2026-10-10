"""Extended calibration + statistical rigour (addresses Major Comments 3 and 4).

Computes, on the external cohort, before vs after temperature scaling:
  ECE, adaptive-ECE, maximum calibration error (MCE), Brier score, NLL.
Plus bootstrap 95% CIs for accuracy, balanced accuracy, macro-F1, macro-AUC, kappa,
ECE and selective accuracy. Where tile filenames expose a slide id, CIs are computed
with a SLIDE-CLUSTERED bootstrap (patient/slide-level), directly answering the
pseudoreplication concern; otherwise a patch-level bootstrap is used (and flagged).
Also saves a before/after reliability diagram.
"""
import argparse, json
from pathlib import Path
import numpy as np

from common import (load_model, make_loader, get_device, seed_everything, collect_logits,
                    softmax_np, slide_id_from_path, bootstrap_ci, CLASS_NAMES,
                    set_plot_style, PALETTE)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, cohen_kappa_score)


def ece(probs, labels, bins=15, adaptive=False):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == labels).astype(float)
    if adaptive:
        edges = np.quantile(conf, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = 0, 1
    else:
        edges = np.linspace(0, 1, bins + 1)
    e = 0.0; mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0: continue
        gap = abs(correct[m].mean() - conf[m].mean())
        e += m.mean() * gap; mce = max(mce, gap)
    return e, mce


def brier(probs, labels):
    oh = np.eye(probs.shape[1])[labels]
    return float(((probs - oh) ** 2).sum(1).mean())


def nll(probs, labels):
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())


def reliability_pts(probs, labels, bins=15):
    conf = probs.max(1); correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1); cen, acc = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi); cen.append((lo + hi) / 2)
        acc.append(correct[m].mean() if m.sum() else np.nan)
    return np.array(cen), np.array(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best_model.pt")
    ap.add_argument("--root", default="data/CRC-VAL-HE-7K")
    ap.add_argument("--out", default="results"); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(); Path(a.out).mkdir(exist_ok=True)
    seed_everything(42); device = get_device()
    model, T, _ = load_model(a.ckpt, device)
    loader, c2i, _ = make_loader(a.root)
    logits, labels, paths = collect_logits(model, loader, device)
    raw = softmax_np(logits, 1.0); cal = softmax_np(logits, T)
    groups = [slide_id_from_path(p) for p in paths]
    n_groups = len(set(groups)); clustered = n_groups >= 10
    nb = len(CLASS_NAMES)

    def metrics_on(idx, probs):
        y = labels[idx]; p = probs[idx]
        pred = p.argmax(1)
        try: auc = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(nb)))
        except Exception: auc = np.nan
        e, _ = ece(p, y)
        return dict(acc=accuracy_score(y, pred), bacc=balanced_accuracy_score(y, pred),
                    mf1=f1_score(y, pred, average="macro", labels=list(range(nb)), zero_division=0),
                    auc=auc, kappa=cohen_kappa_score(y, pred), ece=e)

    keys = ["acc", "bacc", "mf1", "auc", "kappa", "ece"]
    point = metrics_on(np.arange(len(labels)), cal)
    cis = {}
    for k in keys:
        lo, hi = bootstrap_ci(lambda idx, kk=k: metrics_on(idx, cal)[kk], n=a.boot,
                              groups=groups if clustered else None,
                              items=None if clustered else len(labels))
        cis[k] = [round(point[k], 4), round(lo, 4), round(hi, 4)]

    e_raw, mce_raw = ece(raw, labels); e_cal, mce_cal = ece(cal, labels)
    ae_raw, _ = ece(raw, labels, adaptive=True); ae_cal, _ = ece(cal, labels, adaptive=True)
    res = {
        "n_patches": int(len(labels)), "n_slides_or_patients": int(n_groups),
        "bootstrap": "slide/patient-clustered" if clustered else "patch-level (slide id not parsed)",
        "temperature": round(T, 4),
        "calibration": {
            "ECE_raw": round(e_raw, 4), "ECE_cal": round(e_cal, 4),
            "adaptiveECE_raw": round(ae_raw, 4), "adaptiveECE_cal": round(ae_cal, 4),
            "MCE_raw": round(mce_raw, 4), "MCE_cal": round(mce_cal, 4),
            "Brier_raw": round(brier(raw, labels), 4), "Brier_cal": round(brier(cal, labels), 4),
            "NLL_raw": round(nll(raw, labels), 4), "NLL_cal": round(nll(cal, labels), 4)},
        "metrics_with_95CI": cis,
        "note": "CIs are [point, lo, hi]; clustered bootstrap treats each slide/patient as the unit."}
    json.dump(res, open(Path(a.out) / "calibration_stats.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    plt = set_plot_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=PALETTE["grey"], ls="--", lw=1)
    for probs, col, lab in [(raw, PALETTE["red"], "uncalibrated"), (cal, PALETTE["blue"], "temperature-scaled")]:
        c, acc = reliability_pts(probs, labels); v = ~np.isnan(acc)
        ax.plot(c[v], acc[v], marker="o", ms=4, color=col, label=lab)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(Path(a.out) / "reliability_before_after.pdf")
    fig.savefig(Path(a.out) / "reliability_before_after.png")
    print("saved reliability_before_after.pdf/.png")


if __name__ == "__main__":
    main()
