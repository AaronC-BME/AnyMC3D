# AnyMC3D

This repository aims to reproduce [CVPR2026 AnyMC3D](https://arxiv.org/abs/2512.12887) for 3D medical image classification by fine-tuning 2D foundation models. 


---

## How it works

AnyMC3D wraps a frozen 2D / video foundation model with lightweight LoRA adapters. Two backbones are supported:

**DINOv2 (default).** For each 3D volume:

1. All 2D slices are extracted along the chosen axis (axial by default)
2. Each slice is encoded with the LoRA-adapted DINOv2 (ViT-B/14 or ViT-L/14)
3. An attention pooling module aggregates slice embeddings into a single volume embedding
4. A linear head produces the final class prediction

**V-JEPA 2.1 (alternative).** Treats the slice axis natively as a temporal dimension, taking advantage of V-JEPA 2.1's spatiotemporal pretraining and 3D-RoPE positional encoding. For each 3D volume:

1. `num_frames` slices are uniformly sampled along the chosen axis (must be even — V-JEPA's tubelet size is 2)
2. Slices are resized to the model's native crop size (384 for ViT-B), grayscale-replicated to 3 channels, and ImageNet-normalized
3. The frozen + LoRA-adapted V-JEPA encoder produces a `(T', H'·W', D)` token grid where `T' = num_frames / 2` time-tubes and `H'·W' = (crop / 16)²` spatial patches per tube
4. Two-stage pooling — first spatial (within each time-tube), then temporal (across time-tubes) — yields a single volume embedding
5. A linear head produces the final class prediction

---

## Project structure

```
AnyMC3D/
├── train.py                        # Hydra-driven training script
├── inference_online.py             # Legacy inference (use inference_nifti.py instead)
├── inference_nifti.py              # Inference on raw NIfTI (full preprocessing + metrics + plots)
├── balanced_accuracy.py            # Custom balanced accuracy metric
├── preprocess.py                   # Preprocessing pipeline (crop → normalize → resample)
├── pyproject.toml                  # Project dependencies (uv / pip)
├── uv.lock                         # Locked dependency versions
│
├── model_arch/
│   ├── __init__.py
│   ├── anymc3d.py                  # DINOv2-based model + Lightning module
│   └── vjepa2_anymc3d.py          # V-JEPA 2.1-based model + Lightning module
│
├── data_modules/
│   ├── __init__.py
│   ├── pdcad_dataset.py            # PDCAD Dataset and DataModule
│   └── data_augmentation.py        # MONAI augmentation pipeline
│
├── configs/
│   ├── train.yaml                  # Top-level training config (defaults list)
│   ├── data/
│   │   └── pdcad.yaml              # PDCAD data config
│   └── model/
│       ├── anymc3d_vitb.yaml       # DINOv2 ViT-B/14 model config for PDCAD
│       └── vjepa21_vitb.yaml       # V-JEPA 2.1 ViT-B model config for PDCAD
│
└── checkpoints/                    # Model checkpoints (per run, gitignored)
```

---

## Dataset Download

### CT-RATE

CT-RATE is a publicly available chest CT dataset from Istanbul Medipol University. It comprises **50,188 non-contrast chest CT volumes** (25,692 unique scans expanded through multiple reconstructions) from **21,304 unique patients**, paired with radiology reports and multi-abnormality labels for 18 chest findings.

> 🤗 **[Request access on Hugging Face](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)**

Access requires agreeing to the dataset terms (academic/research use only). After approval, volumes and metadata can be downloaded directly from the Hugging Face repository.

A **72-volume subset** for quick setup and testing is available on Google Drive. It contains raw `.nii.gz` volumes (at least 3 per class), labels CSVs, and a splits file (54 train / 18 val) — no preprocessing required before running `preprocess.py`.

> 📁 **[Download CT-RATE subset (Google Drive)](https://drive.google.com/drive/folders/1cx9GDRR-0nj-55OBZgFLQ6ToE8EVr8K8)**

**Task:** 18-class multi-label binary classification (one binary label per abnormality per volume). The 18 abnormalities are:

> Medical material, arterial wall calcification, cardiomegaly, pericardial effusion, coronary artery wall calcification, hiatal hernia, lymphadenopathy, emphysema, atelectasis, lung nodule, lung opacity, pulmonary fibrotic sequela, pleural effusion, mosaic attenuation pattern, peribronchial thickening, consolidation, bronchiectasis, interlobular septal thickening

**Dataset statistics:**

| Split | Patients | Volumes |
|-------|----------|---------|
| Train | 20,000 | 45,149 |
| Val | 1,304 | 2,000 |
| Test | — | 3,039 |
| **Total** | **21,304** | **50,188** |

**Preprocessing:** CT window `[-1000, 400]` (lung window) rescaled to `[0, 1]`, then resized to `476 × 476 × 240`. Three CT windows (all-tissue, soft-tissue, lung: `[-1000, 1000]`, `[-150, 250]`, `[-1000, 400]`) are used as three input channels, following the A.S.L. multi-window scheme described in the paper.

---

## CT-RATE dataset format

### Volume naming convention

Folders are structured as `split_patientID_scanID_reconstructionID`. For example:

```
train_1_a_1.nii.gz     # Training set, patient 1, scan "a", reconstruction 1
valid_53_a_1.nii.gz    # Validation set, patient 53, scan "a", reconstruction 1
```

### Labels CSV

Multi-abnormality labels are provided as a CSV file with one row per volume. The `VolumeName` column matches the volume filename (without extension), and each of the 18 abnormality columns contains a binary value (0 or 1):

```csv
VolumeName,Medical material,Arterial wall calcification,Cardiomegaly,Pericardial effusion,Coronary artery wall calcification,Hiatal hernia,Lymphadenopathy,Emphysema,Atelectasis,Lung nodule,Lung opacity,Pulmonary fibrotic sequela,Pleural effusion,Mosaic attenuation pattern,Peribronchial thickening,Consolidation,Bronchiectasis,Interlobular septal thickening
train_1_a_1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0
train_1_a_2,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0
valid_53_a_1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0
```

Separate CSV files are provided for the train and validation splits:
- `train_labels.csv`
- `valid_labels.csv`

### Splits

**`splits.json`** — defines the train/val/test splits per fold. Two layouts are supported:

*Dict-of-folds* (PDCAD / CT-RATE style):
```json
{
    "0": {
        "train": [
            "CT-RATE_train_8661_a_1",
            "CT-RATE_train_665_a_1",
            "CT-RATE_train_15479_a_1",
            "..."
        ],
        "val": [
            "CT-RATE_train_101_a_1",
            "CT-RATE_train_2275_a_1",
            "CT-RATE_train_6369_a_1",
            "..."
        ]
    }
}
```

Case IDs must match the `.npy` filenames in `data_root` (without the `_0000.npy` suffix). The `test` split is optional — if absent for a given fold, `trainer.test(...)` calls are skipped.

*List-of-folds* :
```json
[
    {"train": [...], "val": [...]},
    {"train": [...], "val": [...]}
]
```

---

## Finetuned Model

A finetuned ViT-B/14 checkpoint on the PDCAD dataset is available on Hugging Face:

> 🤗 **[aaronchoi6/anymc3d-vitb14-pdcad](https://huggingface.co/aaronchoi6/anymc3d-vitb14-pdcad)**

| Detail | Value |
|--------|-------|
| Backbone | DINOv2 ViT-B/14 (frozen) |
| Trainable params | ~455K (LoRA + classifier head) |
| Best val AUROC | 0.90 |
| Training | 150 epochs, fold 0 |

---

## Installation

**Option A: Using uv (recommended)**

```bash
uv sync
```

This creates a `.venv`, resolves all dependencies (including PyTorch with CUDA 12.1), and installs everything in one step.

Activate the environment:
```bash
source .venv/bin/activate
```

**Option B: Using pip**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

---

## Preprocessing

Raw NIfTI volumes are preprocessed with `preprocess.py` (v4) before training. The pipeline applies the following operations in this exact order:

1. **Load** raw `.nii.gz` and reorient to RAS+ canonical orientation; read voxel spacing from the NIfTI header
2. **Crop** to the bounding box of non-zero voxels with a configurable margin (default 4 voxels). Done before normalization so that foreground statistics are not corrupted by large zero-padded background regions.
3. **Normalize** intensities — choice of:
   - `zscore`: foreground-masked z-score via MONAI `NormalizeIntensity(nonzero=True)`. Mean/std are computed on non-zero voxels only; background remains 0. Recommended for structural MRI (T1, T2, FLAIR).
   - `percentile`: `ScaleIntensityRangePercentiles(lower, upper, b_min=0, b_max=1, clip=True)`. Recommended for modalities where extreme intensities carry pathological signal (NM-MRI, PET).

   Done before resampling — resampling first would introduce interpolated values at the foreground/background boundary, corrupting the non-zero mask used by z-score.
4. **Resample** to a target voxel spacing using `scipy.ndimage.zoom` (trilinear, `order=1`). Either:
   - User-supplied via `--target_spacing H,W,S` (mm), or
   - Auto-computed as the per-axis **median spacing** across the dataset (a Pass-1 header scan happens before any volume is processed).
5. **Add channel dim** → `(1, H, W, S)` channel-first for PyTorch
6. **(Optional) Resize** to `target_size³` via MONAI `Resize` (trilinear) if `--target_size` is set
7. **Save** as float32 `.npy`

A `preprocessing_manifest.json` is written to `--output_dir` recording every parameter used (target spacing, normalization mode, crop margin, per-file output shapes, errors), so the run is fully reproducible.

### Usage examples

**Z-score, auto median spacing, no resize** (typical for structural MRI):
```bash
python preprocess.py \
    --input_dir  /data/BMLMPS_FLAIR/imagesTr \
    --output_dir /data/BMLMPS_FLAIR/preprocessed \
    --norm       zscore \
    --workers    4 \
    --verify
```

**Percentile clip (0.5–99.5), fixed 1mm isotropic spacing, resize to 256³** (typical for PDCAD / NM-MRI):
```bash
python preprocess.py \
    --input_dir      /data/PDCAD \
    --output_dir     /data/PDCAD_preprocessed \
    --norm           percentile \
    --lower_pct      0.5 \
    --upper_pct      99.5 \
    --target_spacing 1.0,1.0,1.0 \
    --target_size    256 \
    --workers        4 \
    --verify
```

### Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | *required* | Root folder of raw `.nii.gz` (searched recursively) |
| `--output_dir` | *required* | Destination for `.npy` files and the manifest |
| `--norm` | `zscore` | `zscore` or `percentile` |
| `--lower_pct` / `--upper_pct` | `0.5` / `99.5` | Percentile bounds (only used when `--norm percentile`) |
| `--target_spacing` | `None` (auto median) | Voxel spacing in mm as `H,W,S`, e.g. `1.0,1.0,1.0` |
| `--crop_margin` | `4` | Voxels of margin around the nonzero bounding box |
| `--target_size` | `None` | If set, resize to `target_size³` after resampling |
| `--pattern` | `*.nii.gz` | Glob pattern for input files |
| `--workers` | `4` | Parallel worker processes |
| `--verify` | off | Print stats (shape, min/max/mean/std) for the first 3 outputs |

### Notes per dataset

- **PDCAD**: use `--norm percentile --lower_pct 0 --upper_pct 99.5`. Volumes preserve their original shape `(1, 300, 300, 70)` if you skip `--target_size` and use `--target_spacing` matching the source spacing.
- **CT-RATE**: not handled by this script directly — chest CT requires multi-window ([-1000, 1000], [-150, 250], [-1000, 400]) preprocessing rescaled to [0, 1] per channel (see Dataset section above).

---

## PDCAD dataset format

The dataset loader expects a **flat** directory of preprocessed `.npy` volumes:
```
<data_root>/
    <case_id>_0000.npy     # Preprocessed NM volume (1, H, W, S) float32, values in [0, 1]
    ...
```

Two metadata files are also required:

**Labels** — either JSON or CSV is supported (auto-detected from the file extension of `labels_path`).

*Option A — `labels.json`* (dict mapping case ID to its integer label):
```json
{
    "case_001": 0,
    "case_002": 1,
    "case_003": 0
}
```

*Option B — `labels.csv`* (header row with an identifier column and a label column):
```csv
identifier,label
case_001,0
case_002,1
case_003,0
```

The default column names are `identifier` and `label`. To use different column names, override them in the data config:
```yaml
module:
  labels_path: /path/to/labels.csv
  id_col:      case_id     # default: identifier
  label_col:   class       # default: label
```

For multi-label datasets (e.g. CT-RATE's 18 abnormalities), set `task: multilabel` and `label_cols: [col1, col2, ...]` in the data config — the loader will then read multiple columns per row and return a float vector label per case.

**`splits.json`** — defines the train/val/test splits per fold. Two layouts are supported:

*Dict-of-folds* (PDCAD style):
```json
{
    "0": {
        "train": ["case_001", "case_002", ...],
        "val":   ["case_010", "case_011", ...],
        "test":  ["case_020", "case_021", ...]
    }
}
```

*List-of-folds* (BMLMPS / nnSSL style):
```json
[
    {"train": [...], "val": [...]},
    {"train": [...], "val": [...]}
]
```

The `test` split is optional — if absent for a given fold, `trainer.test(...)` calls are skipped.

---

## Training

### Configuration

All hyperparameters — including optimizer, loss, scheduler, and WandB settings — live in the model config YAML. The top-level `configs/train.yaml` just specifies which data and model configs to use via its defaults list.

Data paths are set in `configs/data/pdcad.yaml`:
```yaml
module:
  data_root:   "/path/to/your/pdcad/data"
  labels_path: "/path/to/your/labels.json"
  splits_path: "/path/to/your/splits.json"
```

### Running training

```bash
# (Optional) set which GPU to use
export CUDA_VISIBLE_DEVICES=0

# DINOv2 ViT-B/14 on PDCAD (recommended starting point)
python train.py data=pdcad model=anymc3d_vitb

# V-JEPA 2.1 ViT-B on PDCAD
python train.py data=pdcad model=vjepa21_vitb

# Background with logging
nohup python train.py data=pdcad model=anymc3d_vitb > logs/anymc3d_vitb_pdcad.log 2>&1 &
```

Any config value can be overridden on the command line:

```bash
# Different fold
python train.py data=pdcad model=anymc3d_vitb data.module.fold=1

# Larger batch size
python train.py data=pdcad model=anymc3d_vitb data.module.batch_size=4

# Multi-fold
python train.py data=pdcad model=anymc3d_vitb 'data.module.fold=[0,1,2]'

# Override a model hyperparameter
python train.py data=pdcad model=anymc3d_vitb model.lora_lr=2e-4
```

Training logs to [Weights & Biases](https://wandb.ai) under the project specified in the model config (`pdcad-anymc3d` by default). Checkpoints are saved under `checkpoints/<run_name>/`.

### Multi-GPU training (DDP)

Distributed Data Parallel (DDP) is enabled by adjusting four parameters in the model config (e.g. `configs/model/anymc3d_vitb.yaml`):

```yaml
# ── Trainer / DDP settings ────────────────────────────────
devices:        1            # int (count), -1 (all visible), or list e.g. [0, 1]
strategy:       auto         # "auto" for single-GPU; "ddp" for multi-GPU
num_nodes:      1            # >1 only for multi-machine training
sync_batchnorm: false        # leave false unless you add BatchNorm layers
```

The defaults (`devices: 1`, `strategy: auto`) preserve single-GPU behavior — DDP only activates when you override them. These knobs live in the model config because run scaling tends to track the model: a smaller backbone may fit on one GPU, while a larger one needs two or more.

**Common launch patterns** (all overridable from the CLI without editing the YAML):

```bash
# 2 GPUs on one machine via DDP
python train.py data=pdcad model=anymc3d_vitb \
    model.devices=2 model.strategy=ddp

# All visible GPUs (use CUDA_VISIBLE_DEVICES to control which)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py data=pdcad model=anymc3d_vitb \
    model.devices=-1 model.strategy=ddp

# Specific GPUs by index
python train.py data=pdcad model=anymc3d_vitb \
    'model.devices=[0,2]' model.strategy=ddp
```

> **Don't use** `python -m torch.distributed.launch` or `torchrun` — Lightning's DDP strategy spawns the worker processes itself when `devices > 1`.

**Things to know before your first DDP run:**

- **Per-GPU batch size scales with rank count.** `data.module.batch_size: 2` on 2 GPUs gives an effective batch of 4. If you want the same effective batch as a single-GPU run, halve the per-GPU batch; if you keep it the same, consider scaling `lora_lr` / `head_lr` proportionally (linear scaling rule).
- **PEFT/LoRA + DDP can raise "parameter unused in forward" errors** because some adapter linears may not be visited every step. If you hit this, switch the strategy:
  ```bash
  python train.py ... model.strategy=ddp_find_unused_parameters_true
  ```
  There's a small perf cost but it always works.
- **`num_workers` is per-rank.** `data.module.num_workers: 8` × 2 GPUs = 16 worker processes. Make sure your machine has the CPU and RAM headroom; otherwise lower it.
- **WandB logs from rank 0 only** — Lightning handles this automatically. Checkpoints, metrics, and console output are also rank-0 only. You'll see one progress bar, not N.
- **Validate on single-GPU first** after any pipeline change before going multi-GPU. Most DDP issues show up as "training works on 1 GPU but breaks on 2".

---

## Inference

Use `inference_nifti.py` to run inference on raw `.nii.gz` files. The script applies the **same preprocessing pipeline used at training time** (load → crop → normalize → resample → optional resize) directly inside the dataset, so volumes fed to the model match what the model saw during training — no need to pre-run `preprocess.py` first.

### Parameter resolution

Preprocessing parameters (normalization, target spacing, crop margin, etc.) are resolved with the following priority chain:

1. **CLI flag** — explicit override (e.g. `--norm percentile`)
2. **`preprocessing_manifest.json`** inside `--run_dir` — auto-copied there by `train.py` at the start of training
3. **`cfg.data` fields** in the saved `config.yaml`
4. **Safe defaults**

For runs trained after the manifest was wired into `train.py`, you can leave every preprocessing flag off and the script will reproduce the training-time preprocessing exactly.

### Running inference

**Default (manifest read from `run_dir`, no extra flags):**
```bash
python inference_nifti.py \
    --run_dir   checkpoints/anymc3d-pdcad-fold0 \
    --nifti_dir /data/PDCAD/raw_nifti
```

**With ground-truth labels (computes metrics + plots):**
```bash
python inference_nifti.py \
    --run_dir    checkpoints/vjepa21-vitb-pdcad-fold0 \
    --nifti_dir  /data/PDCAD/raw_nifti \
    --labels_csv /data/PDCAD/val_cases.csv \
    --checkpoint "epoch=50-val_auroc=0.8808.ckpt"
```

**Manual override (legacy runs without a manifest):**
```bash
python inference_nifti.py \
    --run_dir        checkpoints/anymc3d-pdcad-fold0 \
    --nifti_dir      /data/PDCAD/raw_nifti \
    --norm           percentile \
    --lower_pct      0 --upper_pct 99.5 \
    --target_spacing 1.0,1.0,1.0 \
    --crop_margin    4
```

### Inputs

- `--run_dir` — training run directory containing `config.yaml` and `*.ckpt`. The script auto-selects the checkpoint with the highest `val_auroc=...` in its filename, or you can pass `--checkpoint` to specify one explicitly.
- `--nifti_dir` — directory of raw `.nii` / `.nii.gz` files. Both layouts are supported:
  ```
  flat:    nifti_dir/RJPD_000_0000.nii.gz
  subdir:  nifti_dir/RJPD_000/RJPD_000_0000.nii.gz
  ```
- `--labels_csv` (optional) — CSV with columns `[case_id, label]`. When provided, the script computes AUROC, accuracy, balanced accuracy, F1, and per-class metrics, and writes confusion matrix + ROC plots.

### Outputs

All outputs are written **inside `--run_dir`**:

| File | Description |
|------|-------------|
| `predictions_nifti.csv` | Per-case predictions, per-class probabilities, confidence. Always written. |
| `summary_nifti.csv` | Overall + per-class metrics, plus the resolved preprocessing parameters used. Only when `--labels_csv` is provided. |
| `confusion_matrix_thr0.5.png` | Confusion matrix at the default argmax threshold. Only with labels. |
| `confusion_matrix_thrOpt.png` | Confusion matrix at the Youden's-J optimal threshold (binary only). |
| `roc_curves_nifti.png` | Per-class ROC curves with optimal operating point marked (binary). |

For binary classification, the script automatically computes the **Youden's J optimal threshold** (`max(TPR - FPR)`) and reports metrics under both the default 0.5 threshold and the optimal one, so you can see how much calibration shifts the operating point.

### Batching behavior

When `target_size` is set, all preprocessed volumes share a fixed shape and are batched at `--batch_size` (default 4). When `target_size` is **not** set, cropped/resampled shapes vary per case and the script automatically falls back to per-case inference (`batch_size=1`).

### Inference CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--run_dir` | *required* | Training run directory (contains `config.yaml` + `*.ckpt`) |
| `--nifti_dir` | *required* | Directory of raw NIfTI files |
| `--labels_csv` | `None` | CSV with `[case_id, label]` for metric computation |
| `--checkpoint` | best `val_auroc` | Specific `.ckpt` filename inside `run_dir` |
| `--norm` | from manifest | Override: `zscore`, `percentile`, or `none` |
| `--lower_pct` / `--upper_pct` | from manifest | Override percentile bounds |
| `--target_spacing` | from manifest | Override target spacing as `H,W,S` mm |
| `--crop_margin` | from manifest | Override crop margin in voxels |
| `--target_size` | from manifest | Override final cube resize side length |
| `--no_crop` | off | Disable nonzero-bbox cropping |
| `--no_resample` | off | Disable resampling to target spacing |
| `--class_names` | from `cfg.data` | Override display names for classes |
| `--batch_size` | 4 | DataLoader batch size (used only when `target_size` is set) |
| `--num_workers` | 4 | DataLoader workers |
| `--device` | `cuda` | Inference device |

---

## Key hyperparameters

### DINOv2 backbone (`anymc3d_vitb`)

| Parameter | Default (PDCAD ViT-B) | Description |
|-----------|----------------------|-------------|
| `backbone_name` | `dinov2_vitb14` | DINOv2 variant (`vits14`, `vitb14`, `vitl14`) |
| `lora_rank` | 8 | LoRA rank |
| `lora_alpha` | 16 | LoRA scaling |
| `input_size` | 308 | Slice resize resolution fed to DINOv2 (must be a multiple of 14) |
| `slice_axis` | 3 | Axis to extract 2D slices along (axis 3 = 70 slices for PDCAD) |
| `vision_blocks` | 0 | Number of learnable transformer blocks on top of backbone (0 = disabled) |
| `lora_lr` | 1e-4 | Learning rate for LoRA parameters |
| `head_lr` | 1e-3 | Learning rate for classifier head |
| `lr_scheduler` | `cosine` | LR schedule (`cosine` or `constant`) |
| `focal_gamma` | 2.0 | Focal loss focusing parameter |
| `focal_alpha` | 0.5 | Focal loss class balancing (0.5 for balanced datasets) |
| `max_epochs` | 150 | Maximum training epochs |
| `precision` | `16-mixed` | Mixed precision training |
| `batch_size` | 2 | Training batch size (set in data config) |
| `patch_size` | `[300, 300, 70]` | Input volume dimensions (set in data config) |

### V-JEPA 2.1 backbone (`vjepa21_vitb`)

The V-JEPA 2.1 backbone is loaded via `torch.hub` from the official Meta repo (`facebookresearch/vjepa2`). Both V-JEPA 2 and V-JEPA 2.1 entry points are supported; the `hub_name` parameter selects the variant.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hub_name` | `vjepa2_1_vit_base_384` | torch.hub entry point. See variants below. |
| `vjepa_checkpoint_path` | `null` | Path to pretrained `.pt` weights. Pass at training start; leave `null` for inference (Lightning restores from `.ckpt`). |
| `lora_rank` | 8 | LoRA rank — applied to `qkv` and `proj` linear layers |
| `lora_alpha` | 16 | LoRA scaling |
| `num_frames` | 32 | Number of slices uniformly sampled along `slice_axis`. **Must be even** (tubelet size = 2). |
| `slice_axis` | 3 | Axis to sample frames along (1=H, 2=W, 3=S — use 3 for PDCAD) |
| `dropout` | 0.1 | Dropout before the classifier head |
| `lora_lr` | 1e-4 | Learning rate for LoRA parameters |
| `head_lr` | 1e-3 | Learning rate for classifier head and pooling modules |
| `focal_gamma` | 2.0 | Focal loss focusing parameter |
| `focal_alpha` | 0.25 | Focal loss class balancing |
| `warmup_epochs` | 10 | Linear warmup epochs before cosine annealing |
| `max_epochs` | 150 | Maximum training epochs |
| `task` | `multiclass` | `multiclass` (softmax + CE) or `multilabel` (sigmoid + BCE) |

**Available `hub_name` variants:**

| Variant | Embed dim | Crop size |
|---------|-----------|-----------|
| `vjepa2_vit_large` | 1024 | 256 |
| `vjepa2_vit_huge` | 1280 | 256 |
| `vjepa2_vit_giant` | 1408 | 256 |
| `vjepa2_vit_giant_384` | 1408 | 384 |
| `vjepa2_1_vit_base_384` *(default)* | 768 | 384 |
| `vjepa2_1_vit_large_384` | 1024 | 384 |
| `vjepa2_1_vit_giant_384` | 1408 | 384 |
| `vjepa2_1_vit_gigantic_384` | 1536 | 384 |

`crop_size` is determined by `hub_name` automatically — slices are resized to match before being fed to the encoder. Patch size (16) and tubelet size (2) are constant across all V-JEPA 2 / 2.1 variants.

**Token count formula:**
```
N_tokens = (num_frames / tubelet_size) × (crop_size / patch_size)²
         = T'                          × H'·W'
```
where `T'` is the number of time-tubes (each spans 2 consecutive sampled slices) and `H'·W'` is the spatial grid per tube.

**AnyMC3D-parity flags.** V-JEPA has no CLS token, so pooling has slightly different semantics than the DINOv2 version. All flags default to `False`, giving a flat mean-pool over all tokens — matching the original V-JEPA paper recipe. Toggle them on to mirror DINOv2's `[CLS ; mean(patches)]` aggregation:

| Flag | Default | Description |
|------|---------|-------------|
| `use_25d` | `False` | If `True`, each sampled slice becomes `[s-1, s, s+1]` stacked as 3 channels (neighbours from original slice space, with boundary padding). If `False`, single slice replicated to 3 channels. |
| `use_patch_attn_pool` | `False` | If `True`, AttentionPool over the `H'·W'` spatial tokens per time-tube. If `False`, mean-pool over the spatial grid. |
| `use_patch_concat` | `False` | If `True`, per-time-tube feature = `[attn_pool(spatial) ; mean(spatial)]` → 2D. The V-JEPA analogue of DINOv2's `[CLS ; mean(patches)]`. Implies an internal AttentionPool. |
| `use_slice_attn_pool` | `False` | If `True`, AttentionPool over `T'` time-tubes → volume embedding. If `False`, mean over time-tubes. |

**Checkpoint loading.** `vjepa_checkpoint_path` controls whether pretrained base weights are loaded during `__init__`:
- **Training**: pass the explicit path to the `.pt` file (e.g. `vjepa_2_1_checkpoint/vjepa2_1_vitb_dist_vitG_384.pt`)
- **Inference**: pass `null` (the default) — Lightning's `load_from_checkpoint` restores the full model state from the `.ckpt`, so loading base weights here would be redundant and may fail on machines without the original `.pt` file

---

## Citation

```bibtex
@article{liu2025anymc3d,
  title   = {Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification},
  author  = {Liu et al.},
  journal = {arXiv:2512.12887},
  year    = {2025}
}

@article{hamamci2024generalist,
  title   = {Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography},
  author  = {Hamamci et al.},
  journal = {arXiv:2403.17834},
  year    = {2024}
}
```
