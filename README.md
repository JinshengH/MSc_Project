# From Visual Entailment to Out-of-Context News

Code and analysis outputs for the MSc dissertation **"Contradiction-Aware Fusion for Visual Entailment: A Zero-Shot Cross-Domain Evaluation on Out-of-Context News"**.

Jinsheng Huang, MSc Advanced Computer Science, School of Computer Science, University of Leeds, 2025/26.

## What this project does

Out-of-context misinformation reuses a genuine, unaltered photograph under a story it does not belong to. Because the image itself is real, manipulation forensics has nothing to find: the deception lies in the pairing.

This project asks whether **visual entailment** transfers to that setting without fine-tuning. Visual entailment labels an image and a sentence as entailment, neutral or contradiction (E/N/C). Six model families — including a proposed contradiction-aware cross-attention network (CAMC-Net) — are trained on SNLI-VE with frozen encoders and three seeds, then evaluated **zero-shot** on 500 NewsCLIPpings pairs relabelled by hand for this project.

The relabelling records two layers per pair: the **E/N/C relation**, judged blind from the image and the claim, and the **evidence channel**, the kind of evidence a reader would need to detect the mismatch.

### Main findings

|                                            |                                                        |
| ------------------------------------------ | ------------------------------------------------------ |
| Mismatches decidable from the image alone  | **39.2%** (95% bootstrap CI 33.2–45.2)                 |
| Macro-F1 lost in transfer                  | 0.377 – 0.458 across all families                      |
| Zero-shot macro-F1                         | 0.268 – 0.305, against an all-neutral baseline of 0.22 |
| Visible conflicts missed                   | 75% – 84%                                              |
| Pristine pairs not entailed by their image | 46%                                                    |

What survives the domain shift is not the fusion design but the **alignment of the features beneath it**. Veracity and the image–text relation are distinct properties.

## Repository layout

```
src/                   model definitions, training loop, run configurations
├── train.py           the single training entry point
├── configs/           one module per run (base.py + 23 run configs)
├── models/            CAMC.py, clip_camc.py, clip_fusion.py, late_fusion.py,
│                      text_only.py, image_only.py, ablation.py, encoders.py
└── utils/             env.py, paths.py, plot.py

scripts/               evaluation and analysis entry points (16 scripts)
sbatch/                Slurm job scripts for the Aire HPC facility (24)
analysis/              machine-generated outputs behind every reported number
results/               per-run configuration and training record (23 runs)
jupyter/               exploratory notebooks

annotations_500.csv    the relabelled diagnostic subset (500 rows)
environment.yml        conda environment (Python 3.11, PyTorch, transformers)
setup.sh               environment bootstrap
visualnews_test_files.txt   file list used to fetch the VisualNews subset
```

### `src/`

`train.py` is the only training entry point. Every experiment is selected by naming a **configuration module** rather than by command-line flags, so a run is fully described by one file under `src/configs/`.

`models/` holds one module per model family:

| Module                          | Model in the dissertation                                    |
| ------------------------------- | ------------------------------------------------------------ |
| `CAMC.py`                       | CAMC-Net, the proposed contradiction-aware cross-attention network |
| `clip_camc.py`                  | CLIP-CAMC, the same interaction stack on CLIP features       |
| `clip_fusion.py`                | CLIP fusion, alignment without token-level interaction       |
| `late_fusion.py`                | Difference fusion, the NLI mismatch-feature baseline         |
| `text_only.py`, `image_only.py` | unimodal baselines                                           |
| `ablation.py`                   | variants A1, A3 and A4                                       |
| `encoders.py`                   | the frozen BERT / ViT / CLIP encoders                        |

Two training recipes are used, and `src/configs/base.py` holds the shared defaults:

- **unified** — 5 epochs, weight decay 1e-4. Covers the unimodal baselines, difference fusion and CLIP fusion.
- **regularised** — 8 epochs, dropout 0.2, weight decay 1e-2, label smoothing 0.1, crop augmentation, warmup and cosine decay. Covers CAMC-Net, the ablation variants and CLIP-CAMC. The three-layer depth variant uses this recipe at 6 epochs.

### `scripts/`

| Script                                                       | Purpose                                                |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| `zero_shot_newsclippings.py`                                 | zero-shot evaluation on the diagnostic subset          |
| `eval_snli_ve_test.py`                                       | in-domain evaluation on the SNLI-VE test split         |
| `fsr_full_test.py`                                           | detection metrics on the full NewsCLIPpings test split |
| `make_rq_tables.py`                                          | the research-question tables and figures               |
| `make_results_summary.py`                                    | the unified results table and overview plot            |
| `make_qualitative.py`                                        | selection and rendering of the qualitative cases       |
| `make_annotations_500.py`, `make_annotations_500_html.py`    | build the released subset and its readable view        |
| `make_pilot_annotation_ui.py`                                | the two-stage annotation interface                     |
| `prepare_newsclippings_pilot.py`, `prepare_newsclippings_formal300.py` | sampling for the two annotation batches                |
| `analyze_pilot_zero_shot.py`                                 | pilot-stage analysis                                   |
| `convert_clip_to_safetensors.py`                             | one-off checkpoint conversion                          |
| `make_visualnews_filelist.py`                                | build the VisualNews fetch list                        |
| `prepare_stage0_taxonomy_sample.py`, `label_stage0_preliminary_audit.py` | superseded early-stage tooling, kept for provenance    |

### `analysis/`

Every number, table and figure reported in the dissertation is drawn from this directory.

| Path                               | Contents                                                     |
| ---------------------------------- | ------------------------------------------------------------ |
| `summary.csv`, `summary_pivot.csv` | the unified results table across all checkpoints             |
| `exp4_ablation.csv`                | the ablation comparison                                      |
| `rq/`                              | research-question tables and figures (RQ1 drop, RQ2 distributions, RQ3 channels) |
| `qualitative/`                     | the selected qualitative cases and their rendered panels     |
| `summary_plots/`                   | the in-domain and overview figures                           |
| `fsr_full_test/`                   | per-sample predictions on the full test split                |
| `snli_ve_test_eval/`               | per-sample predictions on the SNLI-VE test split             |

### `results/`

One directory per run, named after its configuration module. Each holds:

- `config.json` — the resolved configuration actually used
- `metrics.csv` — the per-epoch training record

The trained checkpoints themselves are too large to distribute here and are **available on request**.

## Reproducing a result

```bash
conda env create -f environment.yml
conda activate msc

python -m src.train --config camc_reg_seed42     # or any module in src/configs/
python scripts/zero_shot_newsclippings.py
python scripts/make_rq_tables.py
```

Every reported experiment runs from a configuration module, so a result can be reproduced by naming its module, re-running the training entry point, and then the analysis scripts. The frozen encoders and the CLIP checkpoint are downloaded on first use.

## The diagnostic subset

`annotations_500.csv` holds 500 image–text pairs from the NewsCLIPpings test split, one per row, with:

- the identifiers that locate each pair in NewsCLIPpings and VisualNews
- the benchmark's own veracity label and retrieval method
- **`relation_label`** — the E/N/C relation, judged blind from the image and the claim
- **`evidence_channel`** — for falsified pairs, the kind of evidence needed to detect the mismatch
- a confidence grade, a written justification, and the caption text shown to annotator and models

It is released as an **evaluation resource, not a training set**: 500 labels from one annotator are enough to diagnose where models fail, not to supervise a detector.

**No photograph is redistributed.** Access to the images continues to be governed by VisualNews and by the news organisations that own them; the identifier columns point into those sources.

## Data sources

| Resource                                                     | Role here                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| SNLI-VE (Xie et al., 2019)                                   | training split and in-domain evaluation                      |
| e-SNLI-VE (Do et al., 2020), via e-ViL (Kayser et al., 2021) | corrected in-domain evaluation labels                        |
| NewsCLIPpings (Luo et al., 2021)                             | source of the cross-domain evaluation pairs                  |
| VisualNews (Liu et al., 2021)                                | the photographs behind NewsCLIPpings, held locally and never redistributed |

Each release is used on its authors' terms and cited in the dissertation.

## Compute

All training and evaluation ran on **Aire**, the University of Leeds high performance computing facility. `sbatch/` holds the Slurm scripts for each run.
