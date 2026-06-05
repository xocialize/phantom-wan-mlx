# G2 — Substrate readiness (RESOLVED 2026-06-05)

**Gate:** G2 — is the Wan2.1 substrate landed + parity-green so Phantom *consumes* it (not re-ports)?
**Verdict:** ✅ **RESOLVED — reuse `mlx-video`'s `wan_2` module** (NOT LongCat's custom DiT).

## Finding

Phantom-Wan-1.3B is the **stock Wan2.1-T2V-1.3B `WanModel`** (verified `refs/Phantom/phantom_wan/modules/model.py`:
`WanRMSNorm` / `WanSelfAttention(qk_norm + rope_apply)` / `WanT2VCrossAttention` (text-only) / `WanAttentionBlock`
with 6×dim AdaLN modulation). The substrate question is *which existing MLX Wan implementation* to consume.

Surveyed three candidates:

| Candidate | DiT | Verdict |
|---|---|---|
| `longcat-video-mlx` (`longcat_video_dit.py`) | `LongCatVideoTransformer3DModel` — **custom** single-stream blocks + BSA, 13.6B | ❌ not stock Wan2.1 — DiT diverges |
| **`mlx-video` (`models/wan_2/`)** | stock `WanAttentionBlock` (qk_norm, t2v cross-attn, AdaLN 6×dim) | ✅ **exact match** — Bernini-R already reused this (Wan2.2), published |
| `bernini-r-mlx` | wraps `mlx-video` wan_2 | — (use the source directly) |

**`mlx-video` location:** `/Users/dustinnielson/DEV_INT/longcat-avatar-mlx/refs/mlx-video/mlx_video/models/wan_2/`
(editable-installed in sibling venvs). Covers the **entire** Phantom substrate:

| Need | mlx-video `wan_2` file | Note |
|---|---|---|
| DiT | `transformer.py` | stock Wan; configurable dim/ffn/heads/layers → admits 1.3B (1536/8960/12/30) |
| VAE (16-ch) | `vae.py` | z_dim=16 Wan2.1 VAE (CausalConv3d/RMS_norm/Decoder3d). `vae22.py` = 48-ch Wan2.2 (NOT this) |
| Text enc | `text_encoder.py` | umT5-XXL |
| Scheduler | `scheduler.py` → **`FlowUniPCScheduler`** | exactly Phantom's `FlowUniPCMultistepScheduler` (shift=5, 50 steps) |
| 3D RoPE | `rope.py` | stock factorized 3D RoPE |
| Wan weight sanitizer | `convert.py` | for the Phantom_Wan_1.3B checkpoint |

## Consequence for the plan

The port is **substrate-reuse + the G1 4-piece delta**, no DiT re-port:

1. **reference.py** — white-pad (255) aspect-preserving preprocess + VAE-encode K refs → trailing latent frames (G1 §1-2).
2. **input assembly** — `cat([target, refs], axis=1)` (temporal), extended RoPE grid (stock), per-step re-clamp, tail strip (G1 §3, §"re-clamp").
3. **sampling.py** — dual-scale chained CFG (3 fwd/step, zeroed-ref negative, w_img=5/w_text=7.5) over `FlowUniPCScheduler` (G1 §4).
4. **weights.py** — Phantom_Wan_1.3B → mlx-video wan_2 transformer keys (via its `convert.py` sanitizer).

**No new mlx-arsenal primitive. No SA-3D RoPE. No DiT port.** The DiT, VAE, umT5, RoPE, UniPC are all consumed from mlx-video.

## Open / next
- **G3:** pin `bytedance-research/Phantom` 1.3B checkpoint repo + commit; confirm `.pth` vs sharded-safetensors load (subject2video.py:110-132 hardcodes `Phantom_Wan_14B` for the sharded branch → 1.3B likely a single `.pth`).
- Add `mlx-video` as a path/editable dependency in `pyproject.toml`.
- Phase A first action: load Wan2.1-1.3B config into mlx-video's `WanModel`, confirm the Phantom transformer keys map cleanly via `convert.py`, VAE encode/decode parity (should already be green from Bernini-R — if so, that's inherited, not new work).

## Phase A progress (2026-06-05)

**DiT substrate load — ✅ CONFIRMED turnkey.** `mlx_video.models.wan_2.config.WanModelConfig.wan21_t2v_1_3b()`
gives Phantom's exact config (dim1536/ffn8960/12h/30L/t2v/in_dim16). `convert.load_torch_weights` +
`convert.sanitize_wan_transformer_weights` on `Phantom-Wan-1.3B.pth` → **825/826 keys map** into `WanModel`
(the 1 gap is `freqs`, the computed RoPE buffer set at init; load with `strict=False`). Loads in 1.0s,
weights real (blocks.0.self_attn.q mean|w|=0.023), `freqs` populated. → `phantom_wan_mlx/utils/weights.py:load_phantom_dit`.

**Checkpoints pinned (G3 ✅):**
- DiT: `bytedance-research/Phantom` → `Phantom-Wan-1.3B.pth` (5.69 GB, single .pth, stock-Wan keys). → `weights/phantom/`.
- VAE: `Wan-AI/Wan2.1-T2V-1.3B` → `Wan2.1_VAE.pth` (0.51 GB, 16-ch). → `weights/wan-base/`.
- umT5: **reuse** `bernini-r-mlx-weights/ckpt-bf16/t5_encoder.safetensors` (Wan2.2's umt5-xxl == Wan2.1's, already mlx-video fmt) — **skips 11.36 GB download + conversion.**

**mlx-video API confirmed:** `convert.sanitize_wan_{transformer,vae,t5}_weights`; `vae.WanVAE(z_dim=16)`;
`utils.load_vae_{encoder,decoder}(model_path)` (expect pre-converted MLX safetensors); T5 via `AutoTokenizer.from_pretrained("google/umt5-xxl")`.

**Remaining Phase A:** load VAE (`load_wan_vae`) + umT5; Gate A = VAE encode/decode + T5 parity (inherited-green
from Bernini-R/mlx-video — confirm, don't re-port). Then Phase B (reference encode).
Note: `WanModelConfig` default `model_version="2.2"` — `wan21_t2v_1_3b()` sets the right values; verify it
doesn't leave a 2.2-only default (VAE stride etc.) that affects the 2.1 forward.
