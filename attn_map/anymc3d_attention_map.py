"""
anymc3d_attention_map.py — attention-based 3D heatmaps for the AnyMC3D model
============================================================================
Attention only.  For each (case, finding) row of a manifest this writes one
class-agnostic attention volume over the exact preprocessed clip the model saw.

Two variants, selected with --method:

  attn     last-block CLS->patch self-attention  ×  slice-pool attention a[s]

             m_s = (1/h) Σ_j softmax(Q^(L,j) K^(L,j)ᵀ / √d_h)[cls_row, patch_cols]
             M_s = upsample(reshape(m_s, (H/P, W/P)))
             heatmap[:, :, s] = a_s · M_s

           Both factors are attention distributions, so nothing here depends on
           activation magnitudes or gradients.

  rollout  Attention Rollout (Abnar & Zuidema 2020) over all transformer blocks:
           per layer average over heads, add the identity for the residual
           stream, row-normalize, matmul across the stack, take the CLS row.
           Traces influence back to the input patch grid rather than describing
           one layer's token mixing.  Also × a[s].

IMPORTANT — both variants are CLASS-AGNOSTIC.  softmax(q_cls·kᵀ) contains no
class index, so the map is identical for every finding on a given case; the
finding only selects the output directory and the probability in the title.  Use
Grad-CAM if you need class-specific evidence.

Implementation notes:
  * DINOv2 exposes NO eager attention path — Attention.forward calls
    scaled_dot_product_attention and MemEffAttention calls xformers, neither of
    which materializes the post-softmax weights, and XFORMERS_DISABLED only
    swaps one fused kernel for the other.  The weights are therefore re-derived
    from the block's qkv projection.
  * qkv is called through the LoRA-wrapped Linear (LoraConfig targets
    ['qkv','proj','patch_embed.proj']), so the weights are the ADAPTED ones the
    model really used, not the frozen backbone's.
  * For `attn` only the CLS query row is needed -> O(N) instead of O(N²).
    For `rollout` the full head-averaged matrix is needed per layer, accumulated
    one head at a time to keep the peak at O(S·T²) rather than O(S·h·T²).
  * Slices live in the BATCH dimension (a volume is a batch of S images through
    ViT-B), so `act`/`q`/`k` are indexed [S, tokens, dim].

Manifest columns (as produced by saliency_select_cases.py):
  finding, category, identifier, prob, true, thr

Per (case, finding) writes <out_dir>/<finding>/<CATEGORY>/<case_id>/:
  volume.nii.gz            the preprocessed clip (H, W, S)
  saliency_<method>.nii.gz heatmap in [0, 1], same shape
  montage_<method>.png     axial grid, raw over overlay
  slice_attn.png           slice-pool attention across S

Usage:
  python attn_map/anymc3d_attention_map.py \
      --run_dir checkpoints/anymc3d_merlin_liver_multilabel_24cls_vitb_PS384_96slc_150ep_half_fold0 \
      --cases   test_eval/saliency_cases.csv \
      --data_root /data/merlin/preprocessed_merlin_test_liver_half \
      --labels    merlin_test_liver_half_multilabel_labels.csv \
      --splits    /data/merlin/liver_test_half_split.json \
      --out_dir   attn_out/anymc3d --device cuda:0 --method attn rollout
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_modules.cls_data_module import ClassificationDataset
from inference_nifti import find_config, find_best_checkpoint
from inference_test_multilabel import load_model
from compute_saliency import make_main_montage, make_slice_attn_bar, save_nifti

logging.basicConfig(format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("attn-anymc3d")


def collect_blocks(encoder: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """DINOv2 transformer blocks in index order, robust to the PEFT/LoRA nesting
    (the real path is adapted_backbone.base_model.model.blocks.N)."""
    found = []
    pat = re.compile(r"(^|\.)blocks\.(\d+)$")
    for name, mod in encoder.named_modules():
        m = pat.search(name)
        if m:
            found.append((int(m.group(2)), name, mod))
    if not found:
        raise RuntimeError("no '...blocks.<i>' modules found in the encoder")
    found.sort(key=lambda t: t[0])
    return [(n, m) for _, n, m in found]


def cls_attention_row(attn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """(1/h) Σ_j softmax(q_cls k_jᵀ · scale) over patch keys -> [S, N].

    Only query row 0 is computed, so this costs O(N) attention memory instead of
    the O(N²) a full matrix would need.  scale is head_dim**-0.5, matching what
    the fused kernels apply."""
    S, T, D = x.shape
    h = attn.num_heads
    qkv = attn.qkv(x).reshape(S, T, 3, h, D // h)
    q, k = qkv[:, :, 0], qkv[:, :, 1]                    # [S, T, h, hd]
    q_cls = q[:, :1].transpose(1, 2)                     # [S, h, 1, hd]
    k = k.transpose(1, 2)                                # [S, h, T, hd]
    logits = (q_cls * attn.scale) @ k.transpose(-2, -1)  # [S, h, 1, T]
    w = logits.float().softmax(dim=-1)[:, :, 0, 1:]      # drop CLS key -> [S,h,N]
    return w.mean(dim=1)


def headmean_attention(attn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """(1/h) Σ_j softmax(q_j k_jᵀ · scale) -> [S, T, T], one head at a time.

    All heads at once would need O(S·h·T²) (~1.5 GiB at S=96, T=577); this needs
    O(S·T²) (~128 MiB)."""
    S, T, D = x.shape
    h = attn.num_heads
    qkv = attn.qkv(x).reshape(S, T, 3, h, D // h)
    q, k = qkv[:, :, 0], qkv[:, :, 1]
    acc = None
    for j in range(h):
        aj = ((q[:, :, j] * attn.scale) @ k[:, :, j].transpose(-2, -1)).float()
        aj = aj.softmax(dim=-1)
        acc = aj if acc is None else acc.add_(aj)
        del aj
    return acc / h


def residual_normalized(a: torch.Tensor) -> torch.Tensor:
    """Â = row-normalize(A + I) — the residual stream is an identity path."""
    T = a.shape[-1]
    hat = a + torch.eye(T, device=a.device, dtype=a.dtype)
    return hat / hat.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class AttentionHooks:
    """Capture the slice-pool weights, the last block's CLS attention row, and
    (when rollout is requested) every block's residual-normalized matrix."""

    def __init__(self, encoder: torch.nn.Module, want_rollout: bool):
        self.store: dict = {}
        self.hat: dict[int, torch.Tensor] = {}
        self._handles = []

        blocks = collect_blocks(encoder)
        self.n_layers = len(blocks)
        last_name, last_block = blocks[-1]
        log.info(f"CLS-attention block: {last_name}.attn  (of {self.n_layers} blocks)")
        self._handles.append(
            last_block.attn.register_forward_pre_hook(self._cls_hook))

        if want_rollout:
            for layer, (_, blk) in enumerate(blocks):
                self._handles.append(
                    blk.attn.register_forward_pre_hook(self._make_roll_hook(layer)))

        if getattr(encoder, "pool", None) is not None:
            self._handles.append(encoder.pool.register_forward_hook(self._pool_hook))

    def _cls_hook(self, module, inputs):
        with torch.no_grad():
            self.store["cls_attn"] = cls_attention_row(module, inputs[0].detach())

    def _make_roll_hook(self, layer: int):
        def hook(module, inputs):
            with torch.no_grad():
                self.hat[layer] = residual_normalized(
                    headmean_attention(module, inputs[0].detach()))
        return hook

    def _pool_hook(self, module, inputs, output):
        self.store["a"] = output[1].detach()      # AttentionPool -> (v, a); a is [B, S]

    def clear(self):
        self.store.clear()
        self.hat.clear()

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def rollout_cls_row(hat: dict[int, torch.Tensor], n_layers: int) -> torch.Tensor:
    """CLS row of Â_L · … · Â_1 over patch columns -> [S, N]."""
    r = None
    for layer in range(n_layers):
        r = hat[layer] if r is None else hat[layer] @ r
    return r[:, 0, 1:]


def to_grid(flat: torch.Tensor, S: int, side: int, a: torch.Tensor | None) -> torch.Tensor:
    """[S, N] -> [S, side, side] scaled by the slice-pool weights a[s]."""
    g = flat.reshape(S, side, side)
    if a is not None:
        g = g * a.squeeze(0).float().to(g.device)[:, None, None]
    return g


def assemble(grid: torch.Tensor, target_hw: tuple[int, int]) -> np.ndarray:
    """[S, side, side] -> normalized (H, W, S).

    Depth is passed through unchanged (size=(S, H, W) with an S-deep input), so
    only the in-plane side->(H, W) is interpolated — no smearing across slices."""
    S = grid.shape[0]
    H, W = target_hw
    up = F.interpolate(grid[None, None].float(), size=(S, H, W),
                       mode="trilinear", align_corners=False).squeeze()
    up = up.cpu().numpy()
    lo, hi = float(up.min()), float(up.max())
    if hi - lo > 1e-8:
        up = (up - lo) / (hi - lo)
    return np.transpose(up, (1, 2, 0)).astype(np.float32)


def side_from_tokens(n_patch: int) -> int:
    side = int(round(n_patch ** 0.5))
    if side * side != n_patch:
        raise ValueError(f"patch-token count {n_patch} is not a square grid")
    return side


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--ckpt", default=None, type=Path,
                    help="explicit checkpoint; default = best val_auroc in run_dir")
    ap.add_argument("--cases", required=True, type=Path,
                    help="manifest CSV: finding, category, identifier, prob, true, thr")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--method", nargs="+", choices=["attn", "rollout"], default=["attn"])
    ap.add_argument("--findings", nargs="+", default=None,
                    help="restrict the manifest to these findings")
    ap.add_argument("--max_cases", type=int, default=None)
    args = ap.parse_args()

    methods = list(dict.fromkeys(args.method))
    log.info(f"methods: {methods}")

    cfg = OmegaConf.load(find_config(args.run_dir))
    dm = cfg.data.module
    label_cols = list(dm.label_cols)
    ckpt = args.ckpt or find_best_checkpoint(args.run_dir)
    log.info(f"run={args.run_dir.name}  ckpt={Path(ckpt).name}")

    manifest = pd.read_csv(args.cases)
    if args.findings:
        manifest = manifest[manifest["finding"].isin(args.findings)]
        if manifest.empty:
            raise SystemExit("no manifest rows left after --findings filter")
    case_ids = list(dict.fromkeys(manifest["identifier"]))
    if args.max_cases is not None:
        case_ids = case_ids[:args.max_cases]
        manifest = manifest[manifest["identifier"].isin(case_ids)]
    log.info(f"{len(case_ids)} cases, {len(manifest)} (case, finding) rows")

    ds = ClassificationDataset(
        data_root=args.data_root, labels_path=args.labels, splits_path=args.splits,
        split=args.split, fold=dm.get("fold", 0), patch_size=list(dm.patch_size),
        id_col=dm.get("id_col", "identifier"), file_suffix=dm.get("file_suffix", ""),
        task="multilabel", label_cols=label_cols,
        preprocess_strategy=dm.get("preprocess_strategy"),
    )
    id_to_idx = {cid: i for i, cid in enumerate(ds.case_ids)}

    model = load_model(ckpt, cfg)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    modality = model.modalities[0]
    encoder = model.model.encoders[modality]
    hooks = AttentionHooks(encoder, want_rollout="rollout" in methods)

    for cid in case_ids:
        if cid not in id_to_idx:
            log.warning(f"{cid} not in the {args.split} split — skipping")
            continue
        vol, _label, _ = ds[id_to_idx[cid]]
        vol_t = vol.unsqueeze(0).to(device)                  # [1, 1, H, W, S]
        _, _, H, W, S = vol_t.shape
        volume_hws = vol.squeeze(0).cpu().numpy()            # (H, W, S)

        hooks.clear()
        with torch.no_grad():          # attention needs no autograd graph
            logits, _ = model.model({modality: vol_t})       # [1, n_classes]

        a = hooks.store.get("a")
        a_vec = a.squeeze(0).float().cpu().numpy() if a is not None else None
        cls_attn = hooks.store["cls_attn"]                   # [S, N]
        side = side_from_tokens(cls_attn.shape[1])

        grids = {}
        if "attn" in methods:
            grids["attn"] = to_grid(cls_attn, S, side, a)
        if "rollout" in methods:
            grids["rollout"] = to_grid(
                rollout_cls_row(hooks.hat, hooks.n_layers), S, side, a)

        for _, row in manifest[manifest["identifier"] == cid].iterrows():
            finding, category = row["finding"], row["category"]
            fidx = label_cols.index(finding)
            case_out = args.out_dir / finding / category / cid
            case_out.mkdir(parents=True, exist_ok=True)
            prob = float(torch.sigmoid(logits[0, fidx]).cpu())
            title = f"{finding}  {category}  {cid}  p={prob:.3f}  true={int(row['true'])}"

            save_nifti(volume_hws, case_out / "volume.nii.gz")
            for tag, grid in grids.items():
                sal = assemble(grid, (H, W))
                if not np.isfinite(sal).all():
                    raise RuntimeError(f"non-finite values in the {tag} map for {cid}")
                save_nifti(sal, case_out / f"saliency_{tag}.nii.gz")
                make_main_montage(volume_hws, sal, f"{title}  [{tag}]",
                                  case_out / f"montage_{tag}.png")
            if a_vec is not None:
                make_slice_attn_bar(a_vec, title, case_out / "slice_attn.png")
            log.info(f"  {finding:24s} {category}  {cid}  p={prob:.3f} -> {case_out}")

        if torch.cuda.is_available():
            log.info(f"  peak VRAM {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
            torch.cuda.reset_peak_memory_stats(device)

    hooks.close()
    log.info(f"done — outputs under {args.out_dir}")


if __name__ == "__main__":
    main()
