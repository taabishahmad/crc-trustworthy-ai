# Beyond Accuracy: A Reliability, Robustness and Transferability Evaluation of Deep Learning for Colorectal Tissue Classification in Histopathology

## Description

This repository contains the complete code, the trained model checkpoint, and the
experiment outputs for the study *Beyond Accuracy: A Reliability, Robustness and
Transferability Evaluation of Deep Learning for Colorectal Tissue Classification in
Histopathology*.

The project trains an ImageNet-pretrained **EfficientNet-B0** classifier to recognise
**nine colorectal tissue types** in haematoxylin-and-eosin (H&E) histopathology patches,
and then evaluates it far beyond a single accuracy number. Using one trained model, the
code measures six complementary properties, all under a strict **patient-disjoint external
protocol** (trained on one cohort, tested on patients never seen in training):

1. External classification accuracy, with bootstrap confidence intervals.
2. Probability calibration (ECE, adaptive ECE, MCE, Brier score, NLL) with temperature scaling.
3. Predictive uncertainty (Monte Carlo dropout) and selective prediction, compared against
   plain softmax confidence.
4. Grad-CAM explanation faithfulness, audited with random, least-salient and insertion baselines.
5. Robustness to seven controlled image corruptions (stain and scanner variation).
6. Zero-shot transfer to an independent-source colon dataset (LC25000).

The headline finding is deliberately honest: the model is accurate and well calibrated
in-distribution, but it is fragile to stain-colour shift and does not transfer zero-shot to
a dataset acquired elsewhere. Reporting these reliability and robustness properties together
is the main contribution.

## Dataset Information

Three publicly available datasets are used. None are redistributed in this repository; they
are downloaded with the provided script (see Usage).

| Dataset | Role | Patches / images | Source |
|---|---|---|---|
| **NCT-CRC-HE-100K** | Training (development) | 100,000 H&E patches, 224x224, 0.5 um/px, 86 slides | https://doi.org/10.5281/zenodo.1214456 |
| **CRC-VAL-HE-7K** | External test (patient-disjoint) | 7,180 patches from 50 patients, no overlap with training | https://doi.org/10.5281/zenodo.1214456 |
| **LC25000 (colon)** | Zero-shot cross-dataset test | 5,000 adenocarcinoma + 5,000 benign colon images | https://doi.org/10.48550/arXiv.1912.12142 |

The nine tissue classes are: adipose (ADI), background (BACK), debris (DEB), lymphocytes
(LYM), mucus (MUC), smooth muscle (MUS), normal mucosa (NORM), cancer-associated stroma
(STR), and tumour epithelium (TUM).

## Code Information

```
common.py                       shared model, data loading and metric utilities
download_data.py                downloads CRC-VAL-HE-7K and (optionally) LC25000
exp1_faithfulness.py            Grad-CAM faithfulness with random / least-salient / insertion baselines
exp2_calibration_stats.py       calibration metrics + bootstrap confidence intervals + reliability diagram
exp3_uncertainty_ablation.py    softmax vs entropy vs MC-dropout selective prediction
exp4_domain_shift.py            accuracy under seven controlled image corruptions
exp5_cross_dataset.py           zero-shot tumour-vs-benign transfer to LC25000
exp6_backbone_ablation_train.py optional controlled backbone ablation (requires full 100K training set)
best_model.pt                   trained EfficientNet-B0 checkpoint (weights, class map, fitted temperature)
requirements.txt                Python dependencies
results/                        experiment outputs (JSON metrics and PDF/PNG figures)
```

Every evaluation script reuses the single trained checkpoint `best_model.pt`; only the
optional `exp6_backbone_ablation_train.py` retrains a model.

## Requirements

- Python 3.9 or newer
- A GPU is recommended for download/evaluation speed but not required for the light scripts;
  training (exp6) needs a GPU (experiments were run on an NVIDIA T4 via Google Colab).

Python dependencies (also in `requirements.txt`):

```
torch
torchvision
timm
numpy
scipy
scikit-learn
pillow
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage Instructions

1. **Clone and install**

   ```bash
   git clone https://github.com/taabishahmad/crc-trustworthy-ai.git
   cd crc-trustworthy-ai
   pip install -r requirements.txt
   ```

2. **Download the data** (CRC-VAL-HE-7K is ~700 MB)

   ```bash
   python download_data.py --out data
   ```

3. **Run the six evaluations** (each reuses `best_model.pt`; outputs go to `results/`)

   ```bash
   python exp1_faithfulness.py        --ckpt best_model.pt --root data/CRC-VAL-HE-7K --n 200
   python exp2_calibration_stats.py   --ckpt best_model.pt --root data/CRC-VAL-HE-7K
   python exp3_uncertainty_ablation.py --ckpt best_model.pt --root data/CRC-VAL-HE-7K
   python exp4_domain_shift.py        --ckpt best_model.pt --root data/CRC-VAL-HE-7K --subset 3000
   ```

4. **Cross-dataset transfer** (needs a free Kaggle token for LC25000)

   ```bash
   python download_data.py --out data --lc25000
   python exp5_cross_dataset.py --ckpt best_model.pt --root data/lung_colon_image_set/colon_image_sets
   ```

5. **Optional: controlled backbone ablation** (needs the full 100K training set and a GPU)

   ```bash
   python exp6_backbone_ablation_train.py --backbone efficientnet_b0 \
       --root_100k data/NCT-CRC-HE-100K --root_7k data/CRC-VAL-HE-7K
   ```

Each script writes its metrics as JSON and its figures as PDF/PNG into `results/`.

## Methodology

1. **Preprocessing.** Patches are scaled to [0,1] and standardised with ImageNet statistics.
   Training augmentation (random resized crop, flips, mild brightness/contrast/saturation/hue
   jitter) emulates staining and scanner variability; evaluation uses a deterministic centre crop.
2. **Model.** EfficientNet-B0 backbone (all layers fine-tuned) -> global average pooling ->
   dropout (0.3) -> linear(512) + GELU -> dropout -> linear(9) -> softmax.
3. **Training.** Label-smoothed cross-entropy (eps=0.1), AdamW (weight decay 0.05), cosine
   schedule with 2-epoch warmup, peak LR 3e-4, batch size 64, up to 15 epochs with early
   stopping on an 85/15 stratified internal split, mixed precision, fixed seed 42.
4. **Calibration.** Temperature scaling fitted on the validation logits by L-BFGS; reported
   with ECE, adaptive ECE, MCE, Brier and NLL.
5. **Uncertainty / selective prediction.** Monte Carlo dropout (30 passes); selective accuracy
   at several coverages, compared against max-softmax and predictive entropy.
6. **Explanation faithfulness.** Grad-CAM on the final feature block, audited by pixel deletion
   (Grad-CAM vs random vs least-salient order) and an insertion curve.
7. **Robustness.** Accuracy measured under seven corruptions at five severities each.
8. **Transfer.** The nine-class model is applied zero-shot to LC25000 colon, using the TUM
   probability as a tumour score (AUC, plus argmax and Youden-threshold accuracy).

**Key results (external CRC-VAL-HE-7K cohort):** accuracy 0.969 (95% CI 0.965-0.973),
macro-F1 0.960, macro AUC 0.996, Cohen's kappa 0.964; ECE 0.074 -> 0.024 after temperature
scaling; selective accuracy ~0.99 at 90% coverage; zero-shot cross-dataset AUC on LC25000 = 0.542.

## Citations

If you use this code or model, please cite the associated paper:

> Ahmad, T., Saqlain, S. M., AbdElminaam, D. S., Alsekait, D. M., and Gomaa, I. A.
> *Beyond Accuracy: A Reliability, Robustness and Transferability Evaluation of Deep Learning
> for Colorectal Tissue Classification in Histopathology.* (Under review.)

Datasets:

> Kather, J. N., Halama, N., and Marx, A. (2018). *100,000 histological images of human
> colorectal cancer and healthy tissue.* Zenodo. https://doi.org/10.5281/zenodo.1214456

> Borkowski, A. A., et al. (2019). *Lung and Colon Cancer Histopathological Image Dataset
> (LC25000).* arXiv:1912.12142. https://doi.org/10.48550/arXiv.1912.12142

Archived release of this repository: https://doi.org/10.5281/zenodo.23289380

## License and Contribution Guidelines

This project is released under the **MIT License** (see the `LICENSE` file).

Contributions are welcome. Please open an issue to discuss a proposed change, or submit a
pull request with a clear description of what it does and why. For questions about the code
or the study, contact Tabish Ahmad(tabish.bscs4969@student.iiu.edu.pk).
