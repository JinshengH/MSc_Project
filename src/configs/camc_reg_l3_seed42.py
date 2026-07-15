from src.configs.base import Config
from src.utils import paths

# Capacity ablation of camc_reg_seed42: fusion encoder 6 -> 3 layers, same
# regularised recipe. camc_reg still memorises after E4 (train acc 0.956 by
# E8) while val plateaus ~0.70; the frozen encoders cap the extractable
# signal, so surplus fusion capacity only memorises. If 3 layers match or
# beat 0.7023 with a flatter divergence, capacity was the overfit driver.
config = Config(
    exp_name="camc_reg_l3_seed42",
    seed=42,
    data_dir=paths.scratch_project_path(
        "data", "snli-ve", local="jupyter/snli-ve"),
    results_dir=paths.scratch_project_path("results"),
    num_layers=3,
    subset=0,        # full data (train 529k / val 17.9k)
    epochs=6,        # camc_reg peaked at E4; no need for a long tail
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

    main(config, config_name="camc_reg_l3_seed42")
