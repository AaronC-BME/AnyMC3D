# `attn_map/` — attention heatmaps for AnyMC3D and V-JEPA 2.1

Two scripts that turn a trained 3D medical image classifier into an **attention heatmap** you can
open in any NIfTI viewer, plus a PNG montage for quick inspection.

| script | model |
|---|---|
| `anymc3d_attention_map.py` | AnyMC3D (2.5D DINOv2 ViT-B, slice-wise) |
| `vjepa_attention_map.py` | V-JEPA 2.1 (3D ViT-B, spatiotemporal) |

Both write the *same* output layout, so the two models' maps can be compared side by side on
identical cases.

---

## 1. What an "attention heatmap" is here

A transformer decides by mixing information between tokens (small image patches). Every layer
computes, for each token, a probability distribution over all other tokens describing *how much it
draws from each of them*. That distribution is the **attention**.

An attention heatmap answers: **which parts of the image does the representation that feeds the
classifier draw from?**

Two properties are essential to understand before using these maps:

- **They are class-agnostic.** The attention formula contains no class label, so on a given scan the
  map is *identical* for every finding the model predicts. It shows where the model looks, **not why
  it decided a particular diagnosis.** For that you need a class-specific method (Grad-CAM, or the
  exact CAM decomposition available for V-JEPA).
- **They say nothing about correctness.** A visually convincing map can come from a model performing
  at chance level on that finding. Attention is a description of the computation, not evidence that
  the computation is right.

Every map is built from two independent factors:

```
heatmap[:, :, s]  =  (through-plane weight for slice s)  ×  (in-plane map for slice s)
```

The **in-plane** factor says *where within a slice*; the **through-plane** factor says *which slices
matter*. The second comes from the model's own slice-pooling attention `a` in both models. The first
is what differs between the two architectures, and is the substance of Section 3.

---

## 2. Quick start

Run from the repository root, with the project virtualenv:

```bash
# AnyMC3D
.venv/bin/python attn_map/anymc3d_attention_map.py \
    --run_dir   checkpoints/anymc3d_merlin_liver_multilabel_24cls_vitb_PS384_96slc_150ep_half_fold0 \
    --cases     test_eval/saliency_cases.csv \
    --data_root /data/merlin/preprocessed_merlin_test_liver_half \
    --labels    merlin_test_liver_half_multilabel_labels.csv \
    --splits    /data/merlin/liver_test_half_split.json \
    --out_dir   attn_out/anymc3d \
    --device    cuda:0 --method attn

# V-JEPA (same manifest -> directly comparable maps)
.venv/bin/python attn_map/vjepa_attention_map.py \
    --run_dir   checkpoints/vjepa21_merlin_liver_multilabel_24cls_vitb_PS384_96slc_150ep_half_alpha025_fold0 \
    --cases     test_eval/saliency_cases.csv \
    --data_root /data/merlin/preprocessed_merlin_test_liver_half \
    --labels    merlin_test_liver_half_multilabel_labels.csv \
    --splits    /data/merlin/liver_test_half_split.json \
    --out_dir   attn_out/vjepa \
    --device    cuda:0
```

### Inputs

**`--cases`** is a manifest CSV naming which scans to explain. Generate it with the repo's
`saliency_select_cases.py`, which picks one example per confusion-matrix quadrant per finding:

```csv
finding,category,identifier,prob,true,thr
hepatic_lesion,TP,AC423d5df,0.5558,1,0.3134
hepatic_lesion,FP,AC423da17,0.5844,0,0.3134
```

Only `finding`, `category`, `identifier` and `true` are read. `category` is a free-form label used
for the output folder name.

**`--run_dir`** must contain the `config.yaml` frozen at training time. The script reads it to
rebuild the exact preprocessing (patch size, windowing strategy, label columns), so the map is
computed on precisely the voxels the model saw. Without `--ckpt`, the highest-`val_auroc` checkpoint
in the directory is selected automatically and the choice is logged — **check that line**, since it
determines which model you are explaining.

### Common options

| flag | meaning |
|---|---|
| `--method attn rollout` | AnyMC3D only; `attn` is the default, `rollout` is the multi-layer variant |
| `--findings A B` | restrict the manifest to these findings |
| `--max_cases N` | process only the first N scans — use this to smoke-test |
| `--ckpt PATH` | explicit checkpoint instead of best-val auto-selection |
| `--split` | dataset split to load from (default `test`) |
| `--block N` | V-JEPA only; query rows per accumulation block (memory/speed trade-off) |

### Outputs

Per `(scan, finding)`, under `<out_dir>/<finding>/<category>/<identifier>/`:

| file | contents |
|---|---|
| `volume.nii.gz` | the preprocessed clip the model actually saw, `(H, W, S)` |
| `saliency_attn.nii.gz` | the attention heatmap, min-max normalised to `[0, 1]`, same shape |
| `saliency_rollout.nii.gz` | rollout heatmap (AnyMC3D, when requested) |
| `montage_attn.png` | 8 evenly-spaced axial slices: raw CT on top, overlay below |
| `slice_attn.png` | bar chart of the through-plane slice weights |

Open `volume.nii.gz` and `saliency_attn.nii.gz` together in any NIfTI viewer — they share shape and
affine, so the heatmap overlays directly.

**Normalisation caveat:** each map is min-max scaled *within its own volume*. Brightness is therefore
**not comparable between scans** — only the spatial pattern is. To compare magnitudes across scans,
recompute from the raw values rather than the saved NIfTI.

### Cost

Roughly 5–10 s per scan on one modern GPU, dominated by loading the volume from disk. No gradients
are needed, so memory is modest: about **3.7 GiB** peak for AnyMC3D and **2.2 GiB** for V-JEPA at
96 slices of 384×384. Both run on CPU if no GPU is visible, considerably more slowly.

---

## 3. How each script computes the attention

The two models are built differently, and that difference dictates the method.

### 3.1 AnyMC3D — a stack of 2D slices with a CLS token

AnyMC3D treats a volume as `S` independent 2D slices pushed through a DINOv2 ViT-B. Each slice
becomes 1 **CLS token** plus 576 **patch tokens** (a 24×24 grid). The CLS token is a summary token
whose whole job is to aggregate the patch tokens; the classifier head reads only the CLS tokens,
pooled across slices by an attention layer.

That makes the natural question easy to pose: **what did the CLS token attend to?**

For the last transformer block `L`, and each attention head `j`:

```
A[j] = softmax( Q[j] K[j]ᵀ / √d_h )          # attention over [CLS, 576 patches]
m_s  = mean_j  A[j][CLS row, patch columns]  # -> 576 values for slice s
M_s  = reshape(m_s, 24×24) upsampled to (H, W)
heatmap[:, :, s] = a[s] · M_s                # a = slice-pool attention
```

The last block is used because its CLS row is precisely the mechanism by which patch information
reaches the representation the head consumes.

**Why the weights are recomputed rather than read off.** DINOv2 never materialises `A`. Its
`Attention.forward` calls `scaled_dot_product_attention` and `MemEffAttention` calls xformers; both
are fused kernels that produce the attention *output* without ever building the attention *matrix*,
and the `XFORMERS_DISABLED` environment variable only swaps one fused kernel for the other. So the
script re-derives the one row it needs directly from the block's `qkv` projection. Two consequences
worth knowing:

- `qkv` is called through the **LoRA-adapted** layer, so these are the weights the fine-tuned model
  actually used, not the frozen pretrained backbone's.
- Only query row 0 (the CLS row) is computed, which costs `O(N)` memory instead of the `O(N²)` a full
  matrix would need.

Verified: the recomputed row reproduces the module's own attention output to **7e-08**.

**`--method rollout`** adds *Attention Rollout* (Abnar & Zuidema, 2020). Single-layer attention only
describes mixing between the last block's already-heavily-mixed tokens. Rollout instead composes all
12 layers, adding an identity matrix at each one to account for the residual connections that let a
token's own value bypass attention:

```
Â_l = row_normalise(mean_j A_l[j] + I)
R   = Â_12 · Â_11 · … · Â_1        # CLS row of R is the map
```

This traces influence back toward the input patch grid. In practice it suppresses isolated
artifact spikes but blurs localisation, and it correlates with single-layer attention at only
r ≈ 0.26 — the two are not interchangeable. Rollout accumulates each layer's head-average one head
at a time, keeping peak memory at `O(S·T²)` rather than `O(S·h·T²)`.

### 3.2 V-JEPA — a 3D token grid with no CLS token

V-JEPA has **no CLS token**, so there is no "class-to-patch" row to extract. It encodes the whole
volume into a 3D grid of `T' × H' × W'` tokens (27,648 tokens for a 96-slice input) and reaches the
classifier like this:

```
tokens f  ->  spatial MEAN-pool over H'·W'  ->  slice AttentionPool over T' (weights a)  ->  head
```

The substitution used here follows from that path. Because both pooling steps are linear, each token
enters the classifier with an exactly known weight:

```
r_n = a[t(n)] / (H'·W')          # sums to 1 over all tokens
```

`r` **is** the model's own read-out distribution over tokens — the honest stand-in for a CLS query.
The map is the attention each token *receives* from those read-out-weighted queries:

```
m[n] = mean_j  Σ_n'  r[n'] · softmax(q[n'] kᵀ · scale)[j, n]
```

read as: *what does the vector that feeds the classifier attend to?*

**What was deliberately rejected.** A tempting alternative is to use the slice-fusion pool's learned
query `q_t` as a pseudo-CLS query against the last block's keys. That is *not* faithful: `q_t` lives
in the pooled embedding space downstream of the spatial mean, and shares no query/key projection with
the token attention, so the product would be an attention the model never performs. It would look
plausible and mean nothing.

Note also that `a` is applied on the **query** side through `r` and is deliberately *not* applied
again to the keys — doing both would count the slice attention twice.

**Two implementation obstacles, and how they are handled:**

*3D rotary embeddings.* V-JEPA applies RoPE to `q` and `k` **inside** the attention module, and the
upstream source has three different attention classes. Re-deriving `q`/`k` from the projection would
mean replicating that rotation exactly and guessing the right class. Instead the script wraps
`torch.nn.functional.scaled_dot_product_attention` for the duration of the forward pass and keeps the
arguments of the **last** call. Blocks execute in order, so that is the last block, and the captured
tensors are exactly what the model used — rotation, masking and all. The wrapper delegates to the
real kernel, so the forward pass and the predictions are unchanged, and it is restored on exit.

*Sequence length.* With 27,648 tokens a materialised attention matrix is **2.85 GiB per head**, about
34 GiB per layer — so the module's own eager branch (`use_sdpa=False`) is unusable. But `rᵀA` is a
weighted sum over query rows, so it is accumulated over blocks of 256 rows (`--block`), peaking at
`[heads, 256, N]`.

**Self-check.** Because `rᵀA·v` must equal the read-out-weighted attention *output*, the script
asserts this on every scan and aborts on mismatch. It holds to **~2e-06** — which is only possible if
the captured `q`/`k` really are the post-RoPE tensors the model used. If you modify this script, that
assertion is your regression test.

### 3.3 Side-by-side

| | AnyMC3D | V-JEPA 2.1 |
|---|---|---|
| tokenisation | `S` slices × (1 CLS + 576 patches) | one 3D grid, `T'×H'×W'` = 27,648 tokens |
| attention scope | within each slice, independently | global over space **and** time |
| in-plane query | the CLS token | read-out weights `r_n = a[t]/(H'W')` |
| through-plane weight | slice-pool `a[s]`, 96 planes | slice-pool `a[t]`, 48 planes (tubelet 2) |
| weights obtained by | recomputing from LoRA `qkv` | wrapping SDPA to capture post-RoPE `q`/`k` |
| verified to | 7e-08 vs module output | ~2e-06 via `rᵀA·v` identity |
| peak GPU memory | ~3.7 GiB | ~2.2 GiB |

One consequence of the "attention scope" row is worth flagging, because it shows up immediately in
the maps. AnyMC3D normalises attention *within* each slice and then multiplies by `a[s]`, which
downweights uninformative slices — an anatomical prior built into the architecture. V-JEPA
normalises globally across all 27,648 tokens with no per-region renormalisation. Measured on liver
CT, AnyMC3D places ~31% of its attention outside the patient's body versus a **55.6% uniform-map
baseline** (i.e. 0.56× uniform, actively concentrated on anatomy), while V-JEPA places ~58% — about
**1.05× uniform**, meaning its attention carries essentially no body-versus-air preference. V-JEPA's
map is still peaked, but its peaks are not anatomically selective. Treat V-JEPA's raw attention as a
description of the mechanism, not as a localisation of evidence.

---

## 4. Requirements and troubleshooting

Needs the repo root importable (the scripts insert it on `sys.path` themselves, so they run from any
working directory) and the project environment: `torch`, `numpy`, `pandas`, `nibabel`, `matplotlib`,
`omegaconf`, `peft`, `einops`, `lightning`. Backbone weights are fetched via `torch.hub` on first use
and cached under `~/.cache/torch/hub`.

| symptom | cause and fix |
|---|---|
| `<id> not in the test split — skipping` | the manifest names a scan absent from `--splits`; check `--split` and that the manifest matches this dataset |
| `SDPA was never called` (V-JEPA) | the encoder is not taking the fused-attention path, so `q`/`k` could not be captured — check the attention module's `use_sdpa` flag |
| `attention recomputation mismatch` (V-JEPA) | the self-check failed; the captured tensors are not the ones the model used. Do not trust the output — this usually means an upstream change to the attention module |
| `patch-token count is not a square grid` | the backbone's patch grid is non-square; the reshape to `side × side` needs adapting |
| `no '...blocks.<i>' modules found` | the encoder is not the expected DINOv2 block layout |
| CUDA out of memory | lower `--block` (V-JEPA), or pass `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

## 5. Reading the maps responsibly

1. **Class-agnostic.** The same map serves every finding on a scan. Never present it as the reason a
   specific diagnosis was made.
2. **Not a correctness claim.** These maps describe computation, not evidence quality. Measuring
   whether a map is *faithful* requires perturbation testing — deleting the highlighted region and
   confirming the prediction actually collapses (`compare_saliency.py` in the repo root does this).
3. **Brightness is per-scan.** Min-max normalisation is applied within each volume; compare patterns
   between scans, not intensities.
4. **Beware background hotspots.** Vision transformers are known to repurpose a few background tokens
   as global scratch space ("register" or "artifact" tokens). These appear as bright spots in empty
   air at a *fixed grid position* across many slices. They are an artefact of the architecture, not a
   finding. The `rollout` variant suppresses them; single-layer `attn` does not.
5. **Compare models only on identical scans.** Pass the same `--cases` manifest to both scripts.
   Remember the manifest's `category` label reflects whichever model's predictions were used to build
   it, so a case marked FN for one model may be a TP for the other — check each model's own
   probability, printed in the montage title.
