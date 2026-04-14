# Deep Learning Robustness and Adversarial CIFAR-10

This project investigates the robustness of deep neural networks under **distribution shifts** and **adversarial perturbations** on the CIFAR-10 dataset.

We evaluate:
- Robustness to common corruptions (CIFAR-10-C)
- AugMix-based robustness improvement
- Adversarial robustness under PGD attacks
- Knowledge distillation (KD) with robust vs standard teachers
- Adversarial transferability
- Model interpretability via Grad-CAM and t-SNE

---

## 📂 Project Structure

```
.
├── main.py               # Entry point (run experiments)
├── train.py              # Training pipeline
├── test.py               # Evaluation functions
├── parameters.py         # Hyperparameters and configs
├── attack.py             # PGD adversarial attacks
├── augmix.py             # AugMix implementation
├── pretrained.py         # Transfer learning utilities
├── models/               # Model definitions
├── augmix_finetuning_results/   # AugMix evaluation outputs
├── augmix_kd_results/           # KD experiment outputs
├── pgd_transferability_results/ # PGD and transferability outputs
├── best_resnet.pth              # Custom ResNet-18 checkpoint (KD teacher)
├── best_resnet18_option2.pth    # Torchvision ResNet-18 checkpoint (baseline)
└── README.md
```

---

## 🚀 How to Run

### 1. Evaluate CIFAR-10-C robustness (baseline)

```bash
python main.py --mode cifar10c \
    --teacher_path best_resnet18_option2.pth \
    --cifar10c_dir ./CIFAR-10-C
```

### 2. Train AugMix model

```bash
python main.py --mode augmix \
    --teacher_path best_resnet18_option2.pth \
    --save_path best_augmix.pth \
    --cifar10c_dir ./CIFAR-10-C \
    --epochs 20 --lr 1e-4
```

### 3. Knowledge Distillation

```bash
python main.py --mode augmix_kd \
    --epochs 20 \
    --teacher_path best_resnet.pth \
    --augmix_path best_augmix.pth
```

### 4. Adversarial Robustness + Transferability (PGD)

```bash
python main.py --mode pgd \
    --teacher_path best_resnet18_option2.pth \
    --augmix_path best_augmix.pth \
    --results_dir ./pgd_transferability_results \
    --baseline_history history_resnet18_option2.json \
    --augmix_history best_augmix_history.json
```

> **Note:** Run `augmix_kd` before `pgd` to ensure student checkpoints exist for transferability evaluation.

---

## 📊 Results Summary

### Robustness to Corruptions

| Model | Clean Accuracy | mCA |
|---|---|---|
| Baseline ResNet-18 | 0.9264 | 0.726 |
| AugMix ResNet-18 | 0.9416 | 0.865 |

### Knowledge Distillation

| Student | Teacher | Test Accuracy |
|---|---|---|
| SimpleCNN | Baseline ResNet-18 | 0.7621 |
| SimpleCNN | AugMix ResNet-18 | 0.7553 |
| MobileNetV2 | Baseline ResNet-18 | **0.8928** |
| MobileNetV2 | AugMix ResNet-18 | 0.8902 |

### Adversarial Robustness (PGD-20)

| Model | Norm | ε | Clean Acc | Adv Acc |
|---|---|---|---|---|
| Baseline | L∞ | 4/255 | 0.9230 | 0.4564 |
| Baseline | L2 | 0.25 | 0.9230 | 0.7838 |
| AugMix | L∞ | 4/255 | 0.9387 | 0.6240 |
| AugMix | L2 | 0.25 | 0.9387 | 0.8486 |

> **Note:** PGD results are computed on a 5,120-sample subset (40 batches × 128) for computational efficiency.

---

## 📌 Key Findings

- AugMix significantly improves robustness to corruptions (+13.9 pp mCA) and adversarial attacks (+16.8 pp under L∞ PGD-20)
- Robust teachers do not necessarily improve knowledge distillation — the AugMix teacher produces more uniform soft labels, reducing the useful information transferred to the student
- Adversarial examples show low transferability across architectures, with transfer drops of only 0.94–3.81 pp
- There exists a trade-off between robustness and knowledge transfer efficiency

---

## 📁 Data

- CIFAR-10 is automatically downloaded via `torchvision`
- CIFAR-10-C should be downloaded separately from [https://zenodo.org/record/2535967](https://zenodo.org/record/2535967) and placed in `./CIFAR-10-C`

---

## 🔗 Code & Report

Full report and implementation:
👉 [https://github.com/elifizg/deep-learning-robustness-and-adversarial-cifar10](https://github.com/elifizg/deep-learning-robustness-and-adversarial-cifar10)

---

## 📚 References

- He et al., Deep Residual Learning for Image Recognition (CVPR 2016)
- Hendrycks & Dietterich, Benchmarking Neural Network Robustness to Common Corruptions (ICLR 2019)
- Hendrycks et al., AugMix: A Simple Method to Improve Robustness and Uncertainty (ICLR 2020)
- Madry et al., Towards Deep Learning Models Resistant to Adversarial Attacks (ICLR 2018)
- Hinton et al., Distilling the Knowledge in a Neural Network (2015)
