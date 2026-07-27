from src.configs.base import Config
from src.utils import paths

# Exp 4 architecture ablation (guide 5.2), A3: cross-attention only (no diff/product); doubles as the guide 5.1.
# Recipe is byte-identical to camc_reg_seed42 (the A6/full row) so every
# A-row difference is attributable to architecture, not training setup.
config = Config(
    exp_name="ablation_a3_seed42",
    seed=42,
    data_dir=paths.scratch_project_path(
        "data", "snli-ve", local="jupyter/snli-ve"),
    results_dir=paths.scratch_project_path("results"),
    model_name="camc_ablation",
    use_cross_attn=True,
    fusion="it",
    subset=0,        # full data (train 529k / val 17.9k)
    epochs=8,
    batch_size=128,
    lr=1e-4,
    weight_decay=1e-2,
    dropout=0.2,
    label_smoothing=0.1,
    augment_train_images=True,
    scheduler="cosine",
    warmup_ratio=0.05,
)


if __name__ == "__main__":
    from src.train import main

    main(config, config_name="ablation_a3_seed42")
