from src.configs.base import Config
from src.utils import paths

config = Config(
    exp_name="camc_seed42",
    seed=42,
    data_dir=paths.scratch_project_path(
        "data", "snli-ve", local="jupyter/snli-ve"),
    results_dir=paths.scratch_project_path("results"),
    subset=0,       # full data
    epochs=5,
)


if __name__ == "__main__":
    from src.train import main

    main(config, config_name="camc_seed42")
