"""
specimen_invariance_framework/config.py
=======================================
Central configuration dataclasses, hyperparameters, and path definitions for:
- Module 2: Model architectures, conditional adversarial training, and baseline objectives.
- Module 3: Round-robin LOSO, GGSL evaluation, specimen recoverability probe, and calibration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple


@dataclass
class DatasetConfig:
    """Dataset and metadata specifications."""
    # Default root directory for image files and metadata
    data_root: str = "out"
    metadata_csv: str = "out/metadata/metadata.csv"
    label_map_json: str = "out/metadata/label_map.json"
    canonical_split_csv: str = "out/splits/split_canonical.csv"
    
    # Image properties
    image_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Taxonomic cardinality
    num_species: int = 19
    
    # Style randomization settings (Fourier amplitude mixing)
    use_fourier_mixing: bool = True
    fourier_alpha: float = 0.5  # Beta distribution parameter for amplitude interpolation
    fourier_prob: float = 0.5


@dataclass
class ModelConfig:
    """Model architecture and head specifications."""
    # Backbone choices: 'convnext_tiny', 'resnet50', 'tf_efficientnetv2_s', 'swin_tiny_patch4_window7_224'
    backbone_name: str = "convnext_tiny"
    pretrained: bool = True
    embedding_dim: int = 768  # Output dimension of pooling layer (adjusted automatically per backbone)
    projection_dim: int = 256  # Intermediate projection head dimension
    
    # Species classifier head
    species_head_type: str = "linear"  # 'linear', 'arcface'
    arcface_margin: float = 0.50
    arcface_scale: float = 30.0
    
    # Specimen discriminator head (Conditional)
    discriminator_hidden_dim: int = 256
    discriminator_dropout: float = 0.2
    
    # Adversarial schedule lambda_adv(p) = 2 / (1 + exp(-gamma * p)) - 1
    adv_gamma: float = 10.0
    max_lambda_adv: float = 1.0


@dataclass
class TrainingConfig:
    """Optimization and training loop parameters."""
    method: str = "conditional_grl"
    # Supported methods:
    # 1. No intervention: 'ce', 'focal', 'arcface', 'supcon', 'semihard_triplet'
    # 2. General regularization: 'strong_reg', 'mixup'
    # 3. Group robust: 'group_dro', 'irm'
    # 4. Adversarial: 'dann_unconditional', 'conditional_grl' (proposed)
    # 5. Mutual information: 'club'
    # 6. Strong-simple: 'frozen_linear'
    
    epochs: int = 17
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42
    
    # Learning rates: backbone lr < head lr
    lr_backbone: float = 1e-4
    lr_head: float = 5e-4
    weight_decay: float = 1e-2
    min_lr: float = 1e-6
    warmup_epochs: int = 3
    
    # Focal loss parameters
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    
    # Regularization baselines
    label_smoothing: float = 0.1
    dropout_rate: float = 0.3
    mixup_alpha: float = 0.2
    
    # GroupDRO / IRM settings
    group_dro_step_size: float = 0.01
    irm_penalty_weight: float = 1.0
    irm_penalty_anneal_epochs: int = 5
    
    # Checkpointing & logging
    checkpoint_dir: str = "specimen_invariance_outputs/checkpoints"
    logs_dir: str = "specimen_invariance_outputs/logs"
    save_best_only: bool = True
    metric_for_best: str = "macro_f1"  # Evaluated strictly on specimen-disjoint validation


@dataclass
class EvaluationConfig:
    """Module 3 evaluation protocol parameters."""
    # Round-robin LOSO settings
    num_loso_folds: int = 5  # R = 5 to 10 rounds
    
    # Specimen probe settings
    probe_train_ratio: float = 0.5  # 50/50 split of images per specimen
    probe_epochs: int = 25
    probe_lr: float = 1e-3
    
    # Calibration evaluation
    num_calibration_bins: int = 15
    
    # Output paths
    results_dir: str = "specimen_invariance_outputs/eval_results"
    plots_dir: str = "specimen_invariance_outputs/plots"
