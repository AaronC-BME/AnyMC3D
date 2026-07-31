"""
vjepa_attention_map.py — attention-based 3D heatmaps for the V-JEPA 2.1 model
============================================================================
Attention only.  For each (case, finding) row of a manifest this writes one
class-agnostic attention volume over the exact preprocessed clip the model saw.

THE CLS PROBLEM, AND HOW IT IS SOLVED HERE
------------------------------------------
V-JEPA has no CLS token, so "class-to-patch attention" has no direct equivalent
and there is no attention row to read off.  Rather than invent one, we use the
model's OWN read-out distribution over tokens.  With this configuration
(use_patch_concat=False, use_patch_attn_pool=False) the encoder output is

    tokens f  ->  reshape (B, T', H'·W', D)
              ->  spatial MEAN-pool over H'·W'
              ->  slice AttentionPool over T'   (weights a)
              ->  head

so token n enters the classifier with the exact linear weight

    r_n = a_{t(n)} / (H'·W')          (sums to 1)

and the map is the attention received by each key from those weighted queries:

    m[n] = (1/h) Σ_j Σ_{n'} r_{n'} · softmax(q_{n'} kᵀ · scale)[j, n]

i.e. "what does the vector that feeds the classifier attend to" — the CLS-less
analogue of A[cls_row, patch_cols].

Deliberately NOT the slice-fusion query q_t used as a pseudo-CLS query: q_t lives
in the pooled embedding space, shares no QK projection with the token attention,
and would compute an attention the model never performs.

a[t] is applied on the QUERY side through r and is NOT applied again to the keys;
doing both would count the slice attention twice.

IMPORTANT — this map is CLASS-AGNOSTIC.  r is built from a single learned pooling
query shared by every class, and softmax(qkᵀ) has no class index, so the map is
identical for all findings on a given case.

Implementation notes:
  * V-JEPA 2.1 applies 3D RoPE to q and k INSIDE the attention module, and the
    hub source has three attention classes (ACRoPEAttention / RoPEAttention /
    Attention).  Re-deriving q/k from the qkv projection would mean replicating
    rotate_queries_or_keys + separate_positions + grid snapping and guessing the
    class.  Instead we wrap torch.nn.functional.scaled_dot_product_attention for
    the duration of the forward and keep the LAST call's arguments: blocks run in
    order, so that is the last block, and the tensors are exactly what the model
    used.  The wrapper delegates to the real kernel, so the forward is unchanged.
  * The module's own eager branch (use_sdpa=False) is NOT usable: it materializes
    [B, h, N, N], which at N = 27,648 is ~34 GiB.  rᵀA is accumulated over blocks
    of query rows instead, peaking at [h, block, N].
  * Correctness is asserted per case: rᵀA·v must equal the read-out-weighted
    per-head attention output.  That holds only if the captured q/k really are
    the post-RoPE tensors the model used.

Manifest columns (as produced by saliency_select_cases.py):
  finding, category, identifier, prob, true, thr

Per (case, finding) writes <out_dir>/<finding>/<CATEGORY>/<case_id>/:
  volume.nii.gz          the preprocessed clip (H, W, S)
  saliency_attn.nii.gz   heatmap in [0, 1], same shape
  montage_attn.png       axial grid, raw over overlay
  slice_attn.png         slice-pool attention across T'

Usage:
  python attn_map/vjepa_attention_map.py \
      --run_dir checkpoints/vjepa21_merlin_liver_multilabel_24cls_vitb_PS384_96slc_150ep_half_alpha025_fold0 \
      --cases   test_eval/saliency_cases.csv \
      --data_root /data/merlin/preprocessed_merlin_test_liver_half \
      --labels    merlin_test_liver_half_multilabel_labels.csv \
      --splits    /data/merlin/liver_test_half_split.json \
      --out_dir   attn_out/vjepa --device cuda:0
"""
from __future__ import annotations

import argparse
import logging
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
log = logging.getLogger("attn-vjepa")


class SDPACapture:
    """Keep the post-RoPE q/k/v/output of the LAST attention call in a forward.

    Wrapping the kernel rather than the module avoids replicating V-JEPA's 3D
    rotary embedding, which is applied to q and k inside the module and would be
    easy to get subtly wrong.  Restored on exit."""

    def __init__(self) -> None:
        self.q = self.k = self.v = self.out = None
        self._orig = None

    def __enter__(self) -> "SDPACapture":
        self._orig = F.scaled_dot_product_attention

        def wrapper(q, k, v, *args, **kwargs):
            out = self._orig(q, k, v, *args, **kwargs)
            self.q, self.k, self.v, self.out = (
                q.detach(), k.detach(), v.detach(), out.detach())
            return out

        F.scaled_dot_product_attention = wrapper
        return self

    def __exit__(self, *exc) -> None:
        F.scaled_dot_product_attention = self._orig
        self._orig = None


def readout_weights(a: torch.Tensor, t_prime: int, hw_prime: int) -> torch.Tensor:
    """r_n = a_{t(n)} / (H'·W') — each token's exact linear weight into the head."""
    return (a.squeeze(0)[:, None].expand(t_prime, hw_prime) / hw_prime).reshape(-1)


def readout_attention(q: torch.Tensor, k: torch.Tensor, r: torch.Tensor,
                      block: int = 256) -> torch.Tensor:
    """Per-head attention received by each key from the read-out-weighted queries
    -> [h, N].  Accumulated over query-row blocks so no [h, N, N] ever exists.

    scale is head_dim**-0.5 because the model calls SDPA without a scale kwarg,
    which is what SDPA then applies internally."""
    _, h, N, head_dim = q.shape
    scale = head_dim ** -0.5
    k_t = k[0].transpose(-2, -1)                                  # [h, hd, N]
    m = torch.zeros(h, N, device=q.device, dtype=torch.float32)
    for s in range(0, N, block):
        e = min(s + block, N)
        w = ((q[0, :, s:e] * scale) @ k_t).float().softmax(dim=-1)  # [h, blk, N]
        m += (w * r[s:e].view(1, -1, 1)).sum(dim=1)
        del w
    return m


def assemble(grid_thw: torch.Tensor, target_shape: tuple[int, int, int]) -> np.ndarray:
    """(T', H', W') -> normalized (H, W, S) via trilinear upsampling."""
    S, H, W = target_shape
    up = F.interpolate(grid_thw[None, None].float(), size=(S, H, W),
                       mode="trilinear", align_corners=False).squeeze()
    up = up.cpu().numpy()
    lo, hi = float(up.min()), float(up.max())
    if hi - lo > 1e-8:
        up = (up - lo) / (hi - lo)
    return np.transpose(up, (1, 2, 0)).astype(np.float32)


def grid_side(hw_prime: int) -> int:
    side = int(round(hw_prime ** 0.5))
    if side * side != hw_prime:
        raise ValueError(f"H'·W' = {hw_prime} is not a square grid")
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
    ap.add_argument("--block", type=int, default=256,
                    help="query rows per accumulation block (memory/speed knob)")
    ap.add_argument("--findings", nargs="+", default=None,
                    help="restrict the manifest to these findings")
    ap.add_argument("--max_cases", type=int, default=None)
    args = ap.parse_args()

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
    vjepa = model.model
    t_prime = int(vjepa.t_prime)
    hw_prime = int(vjepa.hw_prime)
    side = grid_side(hw_prime)
    log.info(f"token grid: T'={t_prime}, H'=W'={side}  (N={t_prime * hw_prime})")

    slice_pool = getattr(vjepa, "slice_pool", None)
    pool_store: dict = {}
    if slice_pool is not None:
        slice_pool.register_forward_hook(
            lambda m, i, o: pool_store.__setitem__("a", o[1].detach()))

    for cid in case_ids:
        if cid not in id_to_idx:
            log.warning(f"{cid} not in the {args.split} split — skipping")
            continue
        vol, _label, _ = ds[id_to_idx[cid]]
        vol_t = vol.unsqueeze(0).to(device)                  # [1, 1, H, W, S]
        _, _, H, W, S = vol_t.shape
        volume_hws = vol.squeeze(0).cpu().numpy()            # (H, W, S)

        pool_store.clear()
        with torch.no_grad(), SDPACapture() as cap:          # attention needs no graph
            logits = vjepa(vol_t)                            # [1, n_classes]
        if cap.q is None:
            raise RuntimeError("SDPA was never called — the encoder is not using the "
                               "scaled_dot_product_attention path, so post-RoPE q/k "
                               "could not be captured (check the attention use_sdpa flag)")

        a = pool_store.get("a")                              # [1, T']
        a_vec = a.squeeze(0).float().cpu().numpy() if a is not None else None
        n_tok = t_prime * hw_prime
        r = (readout_weights(a.float(), t_prime, hw_prime) if a is not None
             else torch.full((n_tok,), 1.0 / n_tok, device=cap.q.device))
        r = r.to(cap.q.device)

        m = readout_attention(cap.q, cap.k, r, block=args.block)   # [h, N]

        # rᵀA·v must equal the read-out-weighted per-head attention output; this
        # holds only if the captured q/k are the post-RoPE tensors the model used.
        ref = (r.view(1, -1, 1) * cap.out[0].float()).sum(dim=1)
        got = torch.einsum("hn,hnd->hd", m, cap.v[0].float())
        err = (got - ref).abs().max().item()
        if err > 1e-3 * max(ref.abs().max().item(), 1.0):
            raise RuntimeError(f"attention recomputation mismatch for {cid}: "
                               f"max|Δ| = {err:.3e}")
        log.info(f"  {cid}: attention verified, max|rᵀA·v − rᵀSDPA| = {err:.2e}")

        sal = assemble(m.mean(dim=0).reshape(t_prime, side, side), (S, H, W))
        if not np.isfinite(sal).all():
            raise RuntimeError(f"non-finite values in the attention map for {cid}")

        for _, row in manifest[manifest["identifier"] == cid].iterrows():
            finding, category = row["finding"], row["category"]
            fidx = label_cols.index(finding)
            case_out = args.out_dir / finding / category / cid
            case_out.mkdir(parents=True, exist_ok=True)
            prob = float(torch.sigmoid(logits[0, fidx]).cpu())
            title = f"{finding}  {category}  {cid}  p={prob:.3f}  true={int(row['true'])}"

            save_nifti(volume_hws, case_out / "volume.nii.gz")
            save_nifti(sal, case_out / "saliency_attn.nii.gz")
            make_main_montage(volume_hws, sal, f"{title}  [attention]",
                              case_out / "montage_attn.png")
            if a_vec is not None:
                make_slice_attn_bar(a_vec, title, case_out / "slice_attn.png")
            log.info(f"  {finding:24s} {category}  {cid}  p={prob:.3f} -> {case_out}")

        if torch.cuda.is_available():
            log.info(f"  peak VRAM {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
            torch.cuda.reset_peak_memory_stats(device)

    log.info(f"done — outputs under {args.out_dir}")


if __name__ == "__main__":
    main()
