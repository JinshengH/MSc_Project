from dataclasses import dataclass, field
import os
from pathlib import Path

from src.utils import paths


@dataclass
class Config:
    # Experiment
    exp_name: str = "dev"
    seed: int = 42

    # Paths; staged_data_name must match the cp target name in the slurm file
    data_dir: Path = paths.DATA_DIR / "snli-ve"
    results_dir: Path = paths.RESULTS_DIR
    staged_data_name: str = "snli-ve"

    # Model; model_name selects the architecture in src.train.build_model
    model_name: str = "camc"
    hidden_dim: int = 768
    num_heads: int = 8
    num_layers: int = 6
    num_classes: int = 3

    # Preprocessing; defaults reproduce the BERT+ViT experiments. CLIP
    # experiments override both (CLIP tokenizer caps text at 77 tokens and
    # its image normalisation differs from ImageNet's).
    tokenizer_name: str = "bert-base-uncased"
    image_norm: str = "imagenet"    # "imagenet" | "clip"

    # Training
    epochs: int = 2
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0  # gradient clipping threshold
    subset: int = 1000          # first N samples per split; 0 = full data

    # Regularisation; the defaults reproduce the original camc/text_only runs
    dropout: float = 0.1
    label_smoothing: float = 0.0    # train loss only; val loss stays plain CE
    augment_train_images: bool = False  # RandomResizedCrop on train split only

    # LR schedule; "none" = constant lr (original runs), "cosine" = linear
    # warmup over warmup_ratio of total steps then cosine decay to 0
    scheduler: str = "none"
    warmup_ratio: float = 0.0

    # Scheduler-provided machine context; 0 for local runs
    num_workers: int = field(
        default_factory=lambda: int(os.getenv("SLURM_CPUS_PER_TASK", "0"))
    )
