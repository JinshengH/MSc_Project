from src.configs.base import Config
from src.utils import paths

# Seed replicate of clip_camc_seed42 (camc-on-CLIP, the 2x2 missing cell);
# only the seed differs. See clip_camc_seed42.py for the design rationale
# and the recipe note (camc_reg recipe; differs from clip_fusion's unified
# 5-epoch recipe, flag when comparing).
config = Config(
    exp_name="clip_camc_seed43",
    model_name="clip_camc",
    seed=43,
    data_dir=paths.scratch_project_path(
        "data", "snli-ve", local="jupyter/snli-ve"),
    results_dir=paths.scratch_project_path("results"),
    # Local safetensors copy; see scripts/convert_clip_to_safetensors.py
    clip_model_dir=paths.scratch_project_path(
        "models", "clip-vit-base-patch16",
        local="models/clip-vit-base-patch16"),
    tokenizer_name="openai/clip-vit-base-patch16",
    image_norm="clip",
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

    main(config, config_name="clip_camc_seed43")
