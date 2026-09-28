# Specimen-Invariant Wood Identification Framework
## Module 2 (Adversarial Training & Baselines) & Module 3 (Comprehensive Evaluation Protocol)

This directory contains a complete, production-grade implementation of the specimen-invariance research framework designed to eliminate **Specimen-Level Data Leakage (Same-Specimen-Picture Bias - SSPB)** in macroscopic timber identification.

---

### 1. Mathematical Formulation & Architecture

#### Conditional Specimen Discriminator (Proposed)
For single-specimen species, "specimen" and "species" are identical; standard unconditioned GRL cannot eliminate specimen information without erasing species semantics. To overcome this, our discriminator is conditioned on species identity via **masked softmax**:
$$P(s_i \mid y=c, z) = \frac{\exp(w_{c, i}^\top z)}{\sum_{j=1}^{S_c} \exp(w_{c, j}^\top z)}$$

This directly optimizes $I(Z; P \mid Y) \to 0$. Single-specimen taxa ($|G_c| = 1$, such as *Dalbergia cochinchinensis*) are automatically masked out from the adversarial loss branch.

#### Adversarial Loss Objective:
$$\mathcal{L} = \mathcal{L}_{species} + \lambda_{adv}(p)\,\mathcal{L}_{specimen \mid species} \quad (\text{via GRL})$$
$$\lambda_{adv}(p) = \frac{2}{1 + e^{-10p}} - 1, \quad p \in [0, 1]$$
In the codebase, $\mathcal{L}_{specimen \mid species}$ is added with a `+` sign; the **Gradient Reversal Layer (GRL)** automatically inverts the backward gradients ($-\lambda_{adv} \nabla$) toward the visual backbone.

---

### 2. Systematic Baseline Implementations

The framework implements all 6 baseline categories to rigorously validate whether specimen invariance is uniquely achieved by the proposed conditioned GRL:

| Group | Method | Description |
| :--- | :--- | :--- |
| **1. No Intervention** | `focal`, `arcface`, `supcon`, `semihard_triplet` | Standard classification under class imbalance and metric representation learning. |
| **2. General Regularization** | `strong_reg`, `mixup` | Strong weight decay, label smoothing, dropout, and Mixup data augmentation. |
| **3. Adversarial** | `dann_unconditional`, `conditional_grl` | Unconditional DANN predicting global specimen identity vs. proposed species-conditioned GRL. |
| **4. Mutual Information** | `club` | Contrastive Log-ratio Upper Bound (CLUB) direct minimization of mutual information $I(Z; P)$. |
| **5. Strong-Simple** | `frozen_linear` | Frozen ImageNet backbone features + linear classification head. |

---

### 3. Module 3 Evaluation Protocols

1. **Round-Robin Leave-One-Specimen-Out (LOSO)**:
   - Evaluated across $R = 5$ to $10$ folds. In each fold, $\approx 1$ physical block per multi-specimen species is held out for testing, 1 for strict validation, and the remainder for training.
   - All comparative baselines are evaluated on the exact same folds for paired statistical testing.
2. **Generalization Gap from Specimen Leakage (GGSL)**:
   $$\text{GGSL}_{\text{acc}} = \text{Acc}_{\text{leaky}} - \text{Acc}_{\text{LOSO}}$$
   $$\text{GGSL}_{\text{macro-f1}} = \text{F1}_{\text{leaky}} - \text{F1}_{\text{LOSO}}$$
   where $\text{Acc}_{\text{leaky}}$ is measured on a matched stratified image-level split of identical size.
3. **Specimen Recoverability Index (SRI)**:
   - Visual backbone features are frozen.
   - Images of each specimen are split **50/50 into probe-train and probe-test**.
   - A linear probe classifies specimen identity within each species.
   - Accuracy is normalized against random guessing chance ($1 / |G_c|$):
     $$\text{SRI}_c = \frac{\text{Acc}_{probe, c} - 1/|G_c|}{1 - 1/|G_c|}$$
   - Mean SRI measures residual shortcut memorization ($0.0 = \text{invariant}$, $1.0 = \text{leaked}$).
4. **Pareto Sweep & Correlation**:
   - Sweeps $\lambda_{adv} \in [0.0, 2.0]$.
   - Generates Pareto frontier between Strict Accuracy and SRI.
   - Computes linear correlation between SRI and GGSL.
5. **Model Calibration**:
   - Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) across 15 confidence bins.
   - Generates publication-ready Reliability Diagrams.

---

### 4. Directory Layout

```
specimen_invariance_framework/
├── config.py                 # Hyperparameter dataclasses and experiment settings
├── datasets/
│   ├── dataset.py            # TimberDataset and taxonomy mapping logic
│   ├── samplers.py           # SpecimenBalancedBatchSampler
│   ├── augmentations.py      # Transforms + Fourier Amplitude Style Randomization
│   └── splits.py             # Round-Robin LOSO & Leaky Stratified split generators
├── models/
│   ├── backbones.py          # ConvNeXt-Tiny, ResNet-50, EfficientNetV2-S, Swin-T
│   ├── grl.py                # Gradient Reversal Layer with dynamic annealing
│   ├── heads.py              # SpeciesClassifier, Conditional & Unconditional Discriminators
│   ├── club.py               # Discrete CLUB mutual information estimator
│   └── full_model.py         # Unified SpecimenInvariantModel
├── losses/
│   ├── species_losses.py     # Focal Loss, Label Smoothing Cross-Entropy
│   ├── contrastive_losses.py # SupCon Loss, Semi-Hard Triplet Loss
│   └── group_robust_losses.py# GroupDRO, IRM losses
├── trainers/
│   ├── base_trainer.py       # Core trainer with strict specimen-disjoint validation
│   └── trainer_adversarial.py# Unified trainer for proposed GRL and all baselines
├── evaluation/
│   ├── specimen_probe.py     # 50/50 per-specimen probe & SRI calculation
│   ├── calibration.py        # ECE, MCE, and Reliability Diagram plotting
│   ├── metrics.py            # GGSL computation & paired fold statistics
│   └── visualizer.py         # Dual t-SNE (Species vs Specimen), Pareto & Correlation plots
├── train.py                  # Standalone CLI training script
├── evaluate.py               # Standalone CLI evaluation script (GGSL, Probe, Calibration, t-SNE)
├── run_pareto_sweep.py       # Standalone CLI Pareto sweep script
├── run_parallel_dispatcher.py# Multi-GPU dynamic task dispatcher (auto-detects GPUs & runs in parallel)
└── scripts/
    ├── 01_run_loso_baselines.sh    # Run all 13 invariance baselines (auto multi-GPU parallel)
    ├── 02_run_ablation_backbones.sh# Run backbone ablations (auto multi-GPU parallel)
    ├── 03_run_pareto_sweep.sh      # Run lambda_adv sweep and Pareto curve
    └── 04_evaluate_all_metrics.sh  # Evaluate checkpoints and generate figures
```

---

### 5. Multi-GPU Parallel Execution Instructions (Run from Terminal)

The framework includes an intelligent **Multi-GPU Parallel Dispatcher** (`run_parallel_dispatcher.py`) that queries PyTorch at runtime (`torch.cuda.device_count()`), detects all available physical GPUs, and allocates a concurrent worker pool. As soon as a GPU finishes a model, it automatically picks up the next task from the queue, maximizing GPU utilization.

#### 1. Run All 13 Baselines in Parallel (Auto-Detects 1, 2, 4, or 8 GPUs):
```bash
cd g:/S3_paper/specimen_invariance_framework
bash scripts/01_run_loso_baselines.sh
# Hoặc chạy trực tiếp qua Python trên Windows CMD / PowerShell:
python run_parallel_dispatcher.py --mode baselines --fold 0 --epochs 17
```

#### 2. Run Backbone Generalization Ablation in Parallel:
```bash
bash scripts/02_run_ablation_backbones.sh
# Hoặc chạy trực tiếp qua Python:
python run_parallel_dispatcher.py --mode backbones --fold 0 --epochs 17
```

#### 3. Run Custom Subset of Methods on Specific GPUs:
```bash
# Chỉ định GPU cụ thể (ví dụ máy có 4 GPU nhưng chỉ dùng GPU 0 và GPU 1):
python run_parallel_dispatcher.py --mode custom --methods conditional_grl focal arcface --gpus 0 1
```

#### 4. Run lambda_adv Sweep & Generate Pareto Curves:
```bash
bash scripts/03_run_pareto_sweep.sh
```

#### 5. Run Full Evaluation, Probe (SRI), Calibration & t-SNE:
```bash
bash scripts/04_evaluate_all_metrics.sh
```
