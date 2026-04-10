# Deep Learning — HW2: Adversarial Robustness & Knowledge Distillation (CIFAR-10)

This repository contains the full implementation for CS515 HW2, covering CIFAR-10-C robustness evaluation, AugMix fine-tuning, PGD adversarial attacks, Grad-CAM / t-SNE visualisation, and knowledge distillation experiments comparing a baseline ResNet-18 teacher against an AugMix-trained teacher.

---

## Repository Structure

```
.
├── main.py                        # Entry point — all experiment modes
├── train.py                       # Training loop, learning rate scheduler
├── test.py                        # Evaluation utilities
├── attack.py                      # PGD attack, Grad-CAM, t-SNE, transferability
├── pretrained.py                  # ResNet-18 Option 2 (torchvision transfer learning)
├── augmix.py                      # AugMix augmentation pipeline and JSD loss
├── parameters.py                  # Argument parser
├── requirements.txt               # Python dependencies
├── models/
│   ├── CNN.py                     # SimpleCNN student architecture
│   ├── mobilenet.py               # MobileNetV2 student architecture
│   └── ResNet.py                  # Custom ResNet-18 (used as KD teacher)
├── best_resnet.pth                # Custom ResNet-18 trained from scratch (KD teacher)
├── best_resnet18_option2.pth      # Torchvision ResNet-18, transfer learning Option 2
├── augmix_finetuning_results/     # CIFAR-10-C evaluation outputs (AugMix model)
├── augmix_kd_results/             # KD experiment outputs and training curves
├── pgd_transferability_results/   # PGD, Grad-CAM, t-SNE, transferability outputs
└── data/                          # CIFAR-10 dataset (auto-downloaded)
```

Other checkpoint files (`best_augmix.pth`, student `.pth` files) are generated locally during training and tracked in the repository.

---

## Experiment Modes

All experiments are run via `main.py` using `--mode`:

### 1. CIFAR-10-C Robustness Baseline

Evaluates the pre-trained ResNet-18 on clean CIFAR-10 and all 19 CIFAR-10-C corruption types across 5 severity levels.

```bash
python main.py --mode cifar10c \
    --teacher_path best_resnet18_option2.pth \
    --cifar10c_dir /path/to/CIFAR-10-C
```

### 2. AugMix Fine-tuning

Fine-tunes the baseline ResNet-18 with the AugMix consistency loss (JSD regularisation).

```bash
python main.py --mode augmix \
    --teacher_path best_resnet18_option2.pth \
    --save_path best_augmix.pth \
    --cifar10c_dir /path/to/CIFAR-10-C \
    --epochs 20 \
    --lr 1e-4
```

### 3. PGD Adversarial Attack + Visualisation + Transferability

Runs PGD-20 under L∞ and L2 norms, generates Grad-CAM and t-SNE plots, and evaluates adversarial transferability from the teacher to all four distilled student models.

```bash
python main.py --mode pgd \
    --teacher_path best_resnet18_option2.pth \
    --augmix_path best_augmix.pth \
    --results_dir ./pgd_transferability_results \
    --baseline_history history_resnet18_option2.json \
    --augmix_history best_augmix_history.json
```

**Note:** Run `--mode augmix_kd` before this to ensure student checkpoints exist for transferability evaluation.

### 4. Knowledge Distillation Comparison

Trains four student models under a controlled setting — identical KD mode (standard KL divergence), hyperparameters, and random seed — varying only the teacher checkpoint.

| Experiment | Student     | Teacher              |
|------------|-------------|----------------------|
| B.3        | SimpleCNN   | ResNet-18 (baseline) |
| B.3A       | SimpleCNN   | ResNet-18 (AugMix)   |
| B.4        | MobileNetV2 | ResNet-18 (baseline) |
| B.4A       | MobileNetV2 | ResNet-18 (AugMix)   |

```bash
python main.py --mode augmix_kd \
    --epochs 20 \
    --teacher_path best_resnet.pth \
    --augmix_path best_augmix.pth
```

---

## Recommended Run Order

```bash
# 1. Baseline robustness
python main.py --mode cifar10c --teacher_path best_resnet18_option2.pth --cifar10c_dir ./CIFAR-10-C

# 2. AugMix fine-tuning
python main.py --mode augmix --teacher_path best_resnet18_option2.pth --save_path best_augmix.pth --cifar10c_dir ./CIFAR-10-C --epochs 20 --lr 1e-4

# 3. KD experiments (produces student checkpoints needed for step 4)
python main.py --mode augmix_kd --epochs 20 --teacher_path best_resnet.pth --augmix_path best_augmix.pth

# 4. PGD + transferability (requires student checkpoints from step 3)
python main.py --mode pgd --teacher_path best_resnet18_option2.pth --augmix_path best_augmix.pth --results_dir ./pgd_transferability_results --baseline_history history_resnet18_option2.json --augmix_history best_augmix_history.json
```

---

## Key Results

| Metric | Baseline ResNet-18 | AugMix ResNet-18 |
|---|---|---|
| Clean test accuracy | 92.64% | 94.16% |
| Mean corruption accuracy (mCA) | 72.6% | 86.5% |
| PGD-20 L∞ adversarial accuracy | 45.64% | 62.40% |
| PGD-20 L2 adversarial accuracy | 78.38% | 84.86% |

**Knowledge Distillation (Standard KD, T=4, α=0.3):**

| Student     | Teacher     | Test Acc |
|-------------|-------------|----------|
| SimpleCNN   | Baseline    | 76.21%   |
| SimpleCNN   | AugMix      | 75.53%   |
| MobileNetV2 | Baseline    | **89.28%** |
| MobileNetV2 | AugMix      | 89.02%   |

---

## Requirements

```
torch
torchvision
numpy
matplotlib
scikit-learn
```

Install via:
```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

Experiments were run on Google Colab with an NVIDIA T4 GPU.

---

## Checkpoints

| File | Description |
|---|---|
| `best_resnet18_option2.pth` | Torchvision ResNet-18, transfer learning Option 2 |
| `best_resnet.pth` | Custom ResNet-18 trained from scratch (KD teacher) |
| `best_augmix.pth` | AugMix fine-tuned ResNet-18 |
| `best_cnn_kd.pth` | SimpleCNN distilled from baseline teacher |
| `best_cnn_kd_augmix.pth` | SimpleCNN distilled from AugMix teacher |
| `best_mobilenet_kd.pth` | MobileNetV2 distilled from baseline teacher |
| `best_mobilenet_kd_augmix.pth` | MobileNetV2 distilled from AugMix teacher |
