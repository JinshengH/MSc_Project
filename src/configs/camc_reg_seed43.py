from src.configs.base import Config
from src.utils import paths

# Anti-overfit run of the proposed model. camc_seed42 overfits from epoch 4
# (train acc 0.81 vs val 0.69 at E5; val loss rebounds 0.728 -> 0.797), so
# this config keeps the recipe identical and only adds regularisation:
# dropout 0.2, AdamW wd 1e-2, label smoothing 0.1, mild train-image crop
# augmentation, warmup+cosine lr decay (small late steps discourage
# memorisation), and 8 epochs (regularised training converges later; best.pt
# tracking acts as early stopping).
config = Config(
    exp_name="camc_reg_seed43",
    seed=43,
    data_dir=paths.scratch_project_path(
        "data", "snli-ve", local="jupyter/snli-ve"),
    results_dir=paths.scratch_project_path("results"),
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

    main(config, config_name="camc_reg_seed43")
