from src.configs.base import Config
from src.utils import paths

# camc-on-CLIP: the missing cell of the 2x2 {encoder family} x {fusion head}
# design. Same frozen CLIP checkpoint as clip_fusion (alignment held
# identical; only variable vs clip_fusion is the token-level interaction
# stack), same interaction head as camc (only variable vs camc_reg is the
# encoder family). Recipe matches camc_reg_seed42 (the proposed model's
# regularised recipe: dropout 0.2, wd 1e-2, label smoothing 0.1, crop
# augmentation, warmup+cosine, 8 epochs) because the head is the same
# overfit-prone cross-attention stack; note when comparing against
# clip_fusion (unified 5-epoch recipe) that the recipes differ, as already
# flagged for A2 vs A1 in the Exp 4 write-up. CLIP preprocessing overrides:
# its own tokenizer (77-token cap) and normalisation stats.
config = Config(
    exp_name="clip_camc_seed42",
    model_name="clip_camc",
    seed=42,
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

    main(config, config_name="clip_camc_seed42")
