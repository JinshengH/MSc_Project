from src.configs.base import Config
from src.utils import paths

# Image-only baseline: mirrors text_only_seed42 (unified recipe). The image
# CLS feature alone carries no hypothesis information, so this measures how
# far class priors + image statistics get on SNLI-VE (image-bias anchor,
# counterpart of the language-bias anchor).
config = Config(
    exp_name="image_only_seed42",
    model_name="image_only",
    seed=42,
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

    main(config, config_name="image_only_seed42")
