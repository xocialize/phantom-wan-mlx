# G3 — Checkpoint target (PINNED)

**Gate:** G3 — Checkpoint target. 1.3B first; confirm we match the 1.3B config and load path.
**Status:** ✅ **PINNED 2026-06-05** (HF API; sandbox blocks HF git/curl, fetched via web tool).

## Phantom-Wan weights — `bytedance-research/Phantom`
- **Commit pin:** `926cb19b8273d3841edcf905ca4ddb57f8e43207` (lastModified 2025-05-27). License **Apache-2.0**, `gated:false`, public.
- **v1 target — 1.3B:** `Phantom-Wan-1.3B.pth` — **single `.pth`, 5,692,970,224 B (≈5.69 GB)**, LFS oid `3a23db1c…`.
  - **Load path:** the `.pth` branch in `subject2video.py:110-114` → `torch.load(map_location=…)` then `model.load_state_dict(state, strict=False)`. **NOT** the sharded-safetensors branch.
  - `strict=False` because the `.pth` is the **full fine-tuned Wan2.1 DiT** state dict (Phantom is a full-model fine-tune, not an adapter) — conversion is a standard Wan key-sanitize, no base-merge needed.
- **Later — 14B:** `Phantom_Wan_14B-0000{1..6}-of-00006.safetensors` (~57 GB total) + `Phantom_Wan_14B.safetensors.index.json`. Sharded; base name `Phantom_Wan_14B` matches the hardcoded `load_custom_sharded_weights(..., 'Phantom_Wan_14B', ...)` (subject2video.py:118-132).

> ⚠️ **1.3B and 14B use different load paths** — the v1 converter must read the **single `.pth`** (torch.load), not the index. Add the sharded path only when 14B is in scope.

## Substrate weights — `Wan-AI/Wan2.1-T2V-1.3B`
- **Commit pin:** `37ec512624d61f7aa208f7ea8140a131f93afc9a` (2025-03-01). Apache-2.0.
- Needed pieces (already covered by the LongCat Wan2.1 substrate — do not re-port):
  - **VAE:** `Wan2.1_VAE.pth` (16-ch, stride 4/8/8)
  - **Text encoder:** `models_t5_umt5-xxl-enc-bf16.pth`
  - **Tokenizer:** `google/umt5-xxl/` (`spiece.model`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`)
  - (`diffusion_pytorch_model.safetensors` + `config.json` = base Wan DiT — **not needed**; the Phantom `.pth` already carries the full DiT.)

## Config match (1.3B oracle) — re-confirm at convert time
From `configs/wan_s2v_1_3B.py` + `shared_config.py` (already in `config.py`): dim 1536 · ffn 8960 · freq 256 · 12 heads · 30 layers · in/out 16 · text_dim 4096 · patch (1,2,2) · qk_norm · cross_attn_norm · eps 1e-6 · `model_type='t2v'` (text-only cross-attn, no CLIP). VAE z_dim 16, stride (4,8,8); text_len 512; num_train_timesteps 1000. Inference: shift 5.0, 50 steps, guide_img 5.0 / guide_text 7.5.

## Download (when ready)
Layout matches `utils/weights.py` (`weights/phantom/`, `weights/wan-base/`; umT5 reused from Bernini-R):
```bash
huggingface-cli download bytedance-research/Phantom Phantom-Wan-1.3B.pth \
  --revision 926cb19b8273d3841edcf905ca4ddb57f8e43207 --local-dir ./weights/phantom
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B Wan2.1_VAE.pth \
  --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a --local-dir ./weights/wan-base
# umT5: reuse bernini-r-mlx-weights/ckpt-bf16/t5_encoder.safetensors (same umt5-xxl) — see weights.py
```

## Parity oracle
Stand up upstream `Phantom_Wan_S2V` (refs/Phantom) on the 1.3B `.pth` as the PyTorch oracle for the phase gates (Phase A–E in the handoff). Reference assets for eval live in the HF repo `assets/ref1.png … ref18.png` (G4 will pick the 2- and 4-subject sets).
