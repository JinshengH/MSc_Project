from src.configs.base import Config
from src.utils import paths

config = Config(
    exp_name="text_only_seed43",
    model_name="text_only",
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

    main(config, config_name="text_only_seed43")
