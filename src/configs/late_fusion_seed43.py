from src.configs.base import Config
from src.utils import paths

# Late fusion baseline: same recipe as text_only_seed42 / camc_seed42
# (unified recipe: full data / bs 128 / lr 1e-4 / clip 1.0 / wd 1e-4 /
# 5 epochs / seed 42). Model contrast vs CAMC: no cross-attention; CLS
# pooling per modality into the same 4-way fusion vector + classifier.
config = Config(
    exp_name="late_fusion_seed43",
    model_name="late_fusion",
    seed=43,
    data_dir=paths.scratch_project_path(
        "data", "snli-ve", local="jupyter/snli-ve"),
    results_dir=paths.scratch_project_path("results"),
    subset=0,        # full data, same as camc_seed42
    epochs=5,
    batch_size=128,
    lr=1e-4,
)


if __name__ == "__main__":
    from src.train import main

    main(config, config_name="late_fusion_seed43")
