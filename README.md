# Beyond Accuracy: Reliability, Robustness and Transferability Evaluation of Deep Learning for Colorectal Tissue Classification

This repository contains the code, trained model and experiment outputs for the paper
*Beyond Accuracy: A Reliability, Robustness and Transferability Evaluation of Deep Learning
for Colorectal Tissue Classification in Histopathology* (Tabish Ahmad, Department of Computer
Science, International Islamic University Islamabad).

## Overview

We evaluate an ImageNet pretrained **EfficientNet-B0** classifier for nine colorectal tissue
classes, trained on **NCT-CRC-HE-100K** and tested on the patient disjoint **CRC-VAL-HE-7K**
cohort, along six axes beyond accuracy:

1. External accuracy with bootstrap confidence intervals.
2. Probability calibration (ECE, adaptive ECE, MCE, Brier, NLL) with temperature scaling.
3. Monte Carlo dropout uncertainty and selective prediction, compared against softmax confidence.
4. Grad-CAM explanation faithfulness with random, least salient and insertion baselines.
5. Robustness to seven controlled image corruptions (stain and scanner variation).
6. Zero shot transfer to the independent source LC25000 colon dataset.

The trained model is reused by every script; only the optional backbone ablation retrains.

## Key results (external cohort, CRC-VAL-HE-7K)

| Metric | Value |
|---|---|
| Accuracy | 0.969 (95% CI 0.965 to 0.973) |
| Macro F1 | 0.960 |
| Macro AUC | 0.996 |
| Cohen's kappa | 0.964 |
| ECE (after temperature scaling) | 0.074 to 0.024 |
| Selective accuracy at 90% coverage | 0.992 |
| Cross dataset AUC (LC25000, zero shot) | 0.542 |

The model is accurate and well calibrated in distribution but fragile to stain colour shift and
does not transfer zero shot to an independent dataset. These honest findings are the main
contribution.

## Repository structure

```
common.py                      shared model, data and metric utilities
download_data.py               fetch CRC-VAL-HE-7K and (optionally) LC25000
exp1_faithfulness.py           Grad-CAM faithfulness with baselines
exp2_calibration_stats.py      calibration metrics + bootstrap confidence intervals
exp3_uncertainty_ablation.py   softmax vs entropy vs MC-dropout selective prediction
exp4_domain_shift.py           corruption robustness
exp5_cross_dataset.py          zero shot transfer to LC25000
exp6_backbone_ablation_train.py  optional controlled backbone ablation (needs full 100K set)
best_model.pt                  trained EfficientNet-B0 checkpoint (class map + temperature)
requirements.txt               Python dependencies
results/                       experiment outputs (JSON metrics and figures)
```

The large image datasets are not included; they are downloaded with `download_data.py`
(see links below) and are excluded from the repository by `.gitignore`.

## Datasets

- NCT-CRC-HE-100K and CRC-VAL-HE-7K: https://doi.org/10.5281/zenodo.1214456
- LC25000 (lung and colon): https://doi.org/10.48550/arXiv.1912.12142

## Environment

Python 3, PyTorch, timm, scikit-learn, scipy, numpy, pillow, matplotlib. Experiments were run
on a single NVIDIA T4 GPU (Google Colab).

```
pip install -r requirements.txt
```

## Reproducing the experiments

```
python download_data.py --out data
python exp1_faithfulness.py        --ckpt best_model.pt --root data/CRC-VAL-HE-7K --n 200
python exp2_calibration_stats.py   --ckpt best_model.pt --root data/CRC-VAL-HE-7K
python exp3_uncertainty_ablation.py --ckpt best_model.pt --root data/CRC-VAL-HE-7K
python exp4_domain_shift.py        --ckpt best_model.pt --root data/CRC-VAL-HE-7K --subset 3000
# second dataset (needs a Kaggle token for LC25000)
python download_data.py --out data --lc25000
python exp5_cross_dataset.py       --ckpt best_model.pt --root data/lung_colon_image_set/colon_image_sets
```

All outputs (JSON metrics and PDF/PNG figures) are written to `results/`.

## Citation

If you use this code or model, please cite the associated paper. A BibTeX entry will be added
on publication.

## License

Released under the MIT License (see `LICENSE`).
