from src.configs.base import Config
from src.utils import paths

# CLIP fusion baseline: same unified recipe as late_fusion_seed42 (full data /
# bs 128 / lr 1e-4 / clip 1.0 / wd 1e-4 / 5 epochs / seed 42). Model contrast
# vs late_fusion: same ViT-B/16 image tower architecture, but contrastively
# aligned CLIP embeddings (projected 512-d, L2-normalised) instead of
# independently pretrained ViT/BERT features. CLIP preprocessing overrides:
# its own tokenizer (77-token cap) and normalisation stats.
config = Config(
    exp_name="clip_fusion_seed44",
    model_name="clip_fusion",
    seed=44,
    data_dir=paths.scratch_project_path(
        "data", "snli-ve", local="jupyter/snli-ve"),
    results_dir=paths.scratch_project_path("results"),
    # Local safetensors copy; see scripts/convert_clip_to_safetensors.py
    clip_model_dir=paths.scratch_project_path(
        "models", "clip-vit-base-patch16",
        local="models/clip-vit-base-patch16"),
    tokenizer_name="openai/clip-vit-base-patch16",
    image_norm="clip",
    subset=0,        # full data, same as camc_seed42
    epochs=5,
    batch_size=128,
    lr=1e-4,
)


if __name__ == "__main__":
    from src.train import main

    main(config, config_name="clip_fusion_seed44")
