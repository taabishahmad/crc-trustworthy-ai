"""Uncertainty-score ablation + risk-coverage with CIs (addresses points 4 and 16).

Compares three uncertainty scores for selective prediction on the external cohort:
  (a) max-softmax confidence (deterministic),
  (b) predictive entropy (deterministic, temperature-scaled),
  (c) Monte-Carlo-dropout predictive entropy (S passes).
Reports selective accuracy at several coverages with bootstrap CIs, and a risk-coverage
plot overlaying the three. This shows whether MC-dropout actually adds information.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

from common import (load_model, make_loader, get_device, seed_everything, collect_logits,
                    softmax_np, slide_id_from_path, bootstrap_ci, set_plot_style, PALETTE)


def enable_dropout(m):
    for mod in m.modules():
        if mod.__class__.__name__.startswith("Dropout"):
            mod.train()


@torch.no_grad()
def mc_entropy(model, loader, device, passes=30):
    model.eval(); enable_dropout(model)
    runs = []
    for t in range(passes):
        ps = []
        for x, y, p in loader:
            ps.append(torch.softmax(model(x.to(device)), 1).cpu())
        runs.append(torch.cat(ps))
    mean = torch.stack(runs).mean(0).numpy()
    ent = -(mean * np.log(mean + 1e-12)).sum(1)
    return mean, ent


def sel_acc_at(cov, score, correct):
    order = np.argsort(score)                      # most confident first
    cutoff = max(1, int(cov * len(score)))
    sel = order[:cutoff]
    return correct[sel].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best_model.pt")
    ap.add_argument("--root", default="data/CRC-VAL-HE-7K")
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(); Path(a.out).mkdir(exist_ok=True)
    seed_everything(42); device = get_device()
    model, T, _ = load_model(a.ckpt, device)
    loader, c2i, _ = make_loader(a.root)
    logits, labels, paths = collect_logits(model, loader, device)
    cal = softmax_np(logits, T)
    correct = (cal.argmax(1) == labels).astype(float)

    conf_score = 1 - cal.max(1)                               # low = confident
    ent_score = -(cal * np.log(cal + 1e-12)).sum(1)
    mc_mean, mc_ent = mc_entropy(model, loader, device, a.passes)
    scores = {"max_softmax": conf_score, "entropy": ent_score, "mc_dropout_entropy": mc_ent}

    covs = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6]
    groups = [slide_id_from_path(p) for p in paths]
    out = {"coverages": covs, "selective_accuracy": {}}
    for name, s in scores.items():
        row = {}
        for cov in covs:
            lo, hi = bootstrap_ci(lambda idx, ss=s, cc=cov: sel_acc_at(cc, ss[idx], correct[idx]),
                                  groups=groups, n=500)
            row[str(cov)] = [round(float(sel_acc_at(cov, s, correct)), 4), round(lo, 4), round(hi, 4)]
        out["selective_accuracy"][name] = row
    out["note"] = "values [point, lo, hi]; slide/patient-clustered bootstrap."
    json.dump(out, open(Path(a.out) / "uncertainty_ablation.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

    plt = set_plot_style()
    fig, ax = plt.subplots(figsize=(5.2, 4))
    grid = np.linspace(0.3, 1.0, 50)
    for name, col in [("max_softmax", PALETTE["grey"]), ("entropy", PALETTE["blue"]),
                      ("mc_dropout_entropy", PALETTE["red"])]:
        s = scores[name]
        ys = [sel_acc_at(c, s, correct) for c in grid]
        ax.plot(grid, ys, color=col, label=name.replace("_", " "))
    ax.set_xlabel("coverage"); ax.set_ylabel("selective accuracy")
    ax.set_xlim(0.3, 1.0); ax.legend(frameon=False, fontsize=9)
    fig.savefig(Path(a.out) / "risk_coverage_compare.pdf")
    fig.savefig(Path(a.out) / "risk_coverage_compare.png")
    print("saved risk_coverage_compare.pdf/.png")


if __name__ == "__main__":
    main()
