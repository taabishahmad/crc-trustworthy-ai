"""Faithfulness with proper baselines (addresses the 'causally grounded' critique).

For a sample of correctly-classified tiles we compute, for the predicted class:
  - Grad-CAM deletion curve           (remove most-salient pixels first)
  - random-order deletion curve       (remove pixels in random order)
  - least-salient-first deletion curve (remove least-salient pixels first)
  - Grad-CAM insertion curve          (add most-salient pixels onto a blurred image)
and summarise each by its area under the curve, with bootstrap 95% CIs.

A faithful map => deletion AUC (Grad-CAM) << deletion AUC (random) and a high
insertion AUC. This turns a single uninterpretable 0.72 into a comparative result.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from common import (load_model, make_loader, get_device, seed_everything,
                    CLASS_NAMES, set_plot_style, PALETTE, bootstrap_ci, MEAN, STD)


def grad_cam(model, x, cls, device):
    x = x.unsqueeze(0).to(device)
    logits = model(x, return_cam=True)
    model.zero_grad(set_to_none=True)
    logits[0, cls].backward()
    g = model._last_feat.grad
    a = model._last_feat.detach()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    return (cam - cam.min()) / (np.ptp(cam) + 1e-8)


@torch.no_grad()
def prob(model, x, cls, device):
    return torch.softmax(model(x.to(device)), 1)[0, cls].item()


def curve(model, img, order, cls, device, steps, mode, baseline):
    x = img.unsqueeze(0).to(device)
    total = order.size
    fr = np.linspace(0, 1, steps + 1)
    out = []
    H, W = img.shape[-2:]
    for f in fr:
        k = int(f * total)
        work = (baseline.clone() if mode == "insert" else x.clone())
        if k > 0:
            idx = order[:k]
            r, c = np.unravel_index(idx, (H, W))
            if mode == "insert":
                work[0, :, r, c] = x[0, :, r, c]
            else:
                work[0, :, r, c] = baseline[0, :, r, c]
        out.append(prob(model, work, cls, device))
    return fr, np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best_model.pt")
    ap.add_argument("--root", default="data/CRC-VAL-HE-7K")
    ap.add_argument("--n", type=int, default=200, help="tiles sampled")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    Path(a.out).mkdir(exist_ok=True)
    seed_everything(42); device = get_device()
    model, _, _ = load_model(a.ckpt, device)
    loader, c2i, samples = make_loader(a.root, batch_size=1, workers=2)

    rng = np.random.default_rng(42)
    idxs = rng.permutation(len(samples))
    blur = None
    del_cam, del_rnd, del_least, ins_cam = [], [], [], []
    cam_curves, rnd_curves, least_curves, ins_curves = [], [], [], []
    used = 0
    mean_t = torch.tensor(MEAN).view(1, 3, 1, 1)
    for i in idxs:
        if used >= a.n: break
        img, y, _ = loader.dataset[i]
        xb = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(xb).argmax(1))
        if pred != y:                                   # audit on correct predictions
            continue
        cam = grad_cam(model, img, pred, device)
        base_mean = img.mean(dim=(1, 2), keepdim=True).expand_as(img).unsqueeze(0).to(device)
        base_blur = torch.from_numpy(
            __import__("scipy.ndimage", fromlist=["gaussian_filter"]).gaussian_filter(
                img.numpy(), sigma=(0, 8, 8))).unsqueeze(0).to(device) if _has_scipy() else base_mean
        order_cam = np.argsort(cam.ravel())[::-1]
        order_least = np.argsort(cam.ravel())
        order_rnd = rng.permutation(cam.size)
        fr, c1 = curve(model, img, order_cam, pred, device, a.steps, "delete", base_mean)
        _, c2 = curve(model, img, order_rnd, pred, device, a.steps, "delete", base_mean)
        _, c3 = curve(model, img, order_least, pred, device, a.steps, "delete", base_mean)
        _, c4 = curve(model, img, order_cam, pred, device, a.steps, "insert", base_blur)
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        trap = lambda c: float(_trapz(c, fr))
        del_cam.append(trap(c1)); del_rnd.append(trap(c2))
        del_least.append(trap(c3)); ins_cam.append(trap(c4))
        cam_curves.append(c1); rnd_curves.append(c2); least_curves.append(c3); ins_curves.append(c4)
        used += 1

    def ci(v):
        v = np.array(v)
        lo, hi = bootstrap_ci(lambda idx: v[idx].mean(), items=len(v))
        return [round(float(v.mean()), 4), round(lo, 4), round(hi, 4)]

    res = {"n_tiles": used,
           "deletion_auc_gradcam": ci(del_cam),
           "deletion_auc_random": ci(del_rnd),
           "deletion_auc_least_salient": ci(del_least),
           "insertion_auc_gradcam": ci(ins_cam),
           "note": "deletion: lower=more faithful; Grad-CAM should be << random. "
                   "insertion: higher=more faithful. [mean, ci_lo, ci_hi]"}
    json.dump(res, open(Path(a.out) / "faithfulness_baselines.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    plt = set_plot_style()
    fig, ax = plt.subplots(figsize=(5.4, 4))
    fr = np.linspace(0, 1, a.steps + 1)
    ax.plot(fr, np.mean(cam_curves, 0), color=PALETTE["red"], label="Grad-CAM (del)")
    ax.plot(fr, np.mean(rnd_curves, 0), color=PALETTE["grey"], label="random (del)")
    ax.plot(fr, np.mean(least_curves, 0), color=PALETTE["blue"], label="least-salient (del)")
    ax.plot(fr, np.mean(ins_curves, 0), color=PALETTE["green"], ls="--", label="Grad-CAM (ins)")
    ax.set_xlabel("fraction of pixels perturbed"); ax.set_ylabel("predicted probability")
    ax.set_xlim(0, 1); ax.legend(frameon=False, fontsize=9)
    fig.savefig(Path(a.out) / "faithfulness_curves.pdf")
    fig.savefig(Path(a.out) / "faithfulness_curves.png")
    print("saved faithfulness_curves.pdf/.png")


def _has_scipy():
    try:
        import scipy.ndimage  # noqa
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
