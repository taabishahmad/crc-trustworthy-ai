# CRC paper — strengthening experiments

These scripts answer the reviewer's five "major comments" by adding real evidence,
reusing your already-trained model (`best_model.pt`). Nothing here changes that model
except the optional backbone-ablation script.

## What each experiment fixes

| Script | Reviewer concern it addresses | Needs |
|---|---|---|
| `exp1_faithfulness.py` | "Grad-CAM not causally grounded" → adds **random / least-salient / insertion baselines** + CIs | model + 7K |
| `exp2_calibration_stats.py` | weak calibration + stats → **Brier, NLL, MCE, adaptive-ECE**, before/after diagram, **slide-clustered bootstrap CIs** | model + 7K |
| `exp3_uncertainty_ablation.py` | "MC-dropout not justified" → **softmax vs entropy vs MC-dropout** selective prediction with CIs | model + 7K |
| `exp4_domain_shift.py` | "no domain-shift experiment" → **stain/scanner corruption robustness curves** | model + 7K |
| `exp5_cross_dataset.py` | "single dataset / not multi-source" → **zero-shot cross-source validation on LC25000 colon** (tumour vs benign) | model + LC25000 |
| `exp6_backbone_ablation_train.py` | "no ablation / can't attribute gain" → **controlled backbone comparison** (EfficientNet-B0, ResNet50, no-label-smoothing) | full 100K + GPU |

Experiments 1–4 use only your 7K external set and the saved model — run these first;
they are cheap (~10–30 min total on a Colab GPU) and already cover 4 of the 5 major
comments. Exp 5 needs one download. Exp 6 is the heavy, optional one.

## Setup (Google Colab, GPU runtime)

```bash
!pip -q install -r requirements.txt
# put best_model.pt in this folder, then:
!python download_data.py --out data            # CRC-VAL-HE-7K (~700 MB)
```

## Run (experiments 1–4: model + 7K only)

```bash
!python exp1_faithfulness.py       --ckpt best_model.pt --root data/CRC-VAL-HE-7K --n 200
!python exp2_calibration_stats.py  --ckpt best_model.pt --root data/CRC-VAL-HE-7K
!python exp3_uncertainty_ablation.py --ckpt best_model.pt --root data/CRC-VAL-HE-7K
!python exp4_domain_shift.py       --ckpt best_model.pt --root data/CRC-VAL-HE-7K --subset 2000
```

## Run (experiment 5: second, independent dataset)

```bash
!pip -q install kaggle        # then upload kaggle.json (Account > Create New Token)
!python download_data.py --out data --lc25000
# arrange data/lc25000_colon/colon_aca and .../colon_n (see download_data.py notes)
!python exp5_cross_dataset.py --ckpt best_model.pt --root data/lc25000_colon
```

## Run (experiment 6: OPTIONAL controlled ablation — needs full 100K training set)

```bash
!python download_data.py --out data            # also grab NCT-CRC-HE-100K zip if training
# for each backbone:
!python exp6_backbone_ablation_train.py --backbone efficientnet_b0 --root_100k data/NCT-CRC-HE-100K --root_7k data/CRC-VAL-HE-7K
!python exp6_backbone_ablation_train.py --backbone resnet50        --root_100k data/NCT-CRC-HE-100K --root_7k data/CRC-VAL-HE-7K
!python exp6_backbone_ablation_train.py --backbone convnext_tiny --label_smoothing 0.0 --root_100k data/NCT-CRC-HE-100K --root_7k data/CRC-VAL-HE-7K
```

## What to send back

Zip the whole `results/` folder (all `*.json` and `*.pdf`) and share it. I will read the
numbers and update the paper + comparison tables for PeerJ Computer Science.

> Tip: if a GPU session is short, run exp1–exp4 first — those four alone materially
> strengthen the paper. exp5 (second dataset) is the single most persuasive addition.
