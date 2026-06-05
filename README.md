# phantom-wan-mlx

Apple MLX port of **[Phantom-video/Phantom](https://github.com/Phantom-video/Phantom)** (Phantom-Wan, ByteDance, ICCV 2025) — **multi-subject subject-to-video (S2V)**: compose up to 4 distinct characters into one shot from reference images. Apache-2.0.

> **Status: scaffold + G1 locked.** No model code yet. The reference-injection mechanism (the only net-new surface) is fully specified in [`_research/G1_INJECTION.md`](_research/G1_INJECTION.md), read against upstream. Port plan: `../XDocs/Phantom-Wan-MLX-Port-Handoff.md`.

## Why this is cheap
Rides the **Wan2.1 substrate** already published for LongCat-Video (Wan VAE 16-ch + umT5-XXL + FlowUniPC + 3D RoPE) — ~75–80% reuse. The DiT forward is **unchanged** stock Wan2.1.

## The only new surface (G1)
1. **Reference encode** — aspect-preserving resize + white-pad → [-1,1] → VAE-encode each ref to one latent frame.
2. **Temporal append** — concat K ref frames at the **tail** of the target latent (`axis=temporal`); refs get ordinary 3D-RoPE positions (no SA-3D). Re-clamp clean refs each step; strip the K-frame tail at the end.
3. **Dual-scale chained CFG** — 3 forwards/step: `neg(zero refs,∅) + w_img·(refs,∅ − neg) + w_text·(refs,text − refs,∅)`, defaults `w_img=5.0`, `w_text=7.5`.

## Substrate / config (1.3B v1 oracle)
Wan2.1-T2V-1.3B: dim 1536 · ffn 8960 · 12 heads · 30 layers · patch (1,2,2) · 16-ch VAE (stride 4/8/8) · umT5-xxl · text-only cross-attn (no CLIP). 14B variant: dim 5120 · ffn 13824 · 40/40.

## Layout
```
phantom_wan_mlx/
  config.py          # S2V 1.3B/14B config (from upstream configs)
  pipeline_mlx.py    # s2v(...) entry  (stub)
  sampling.py        # dual-scale chained CFG loop  (stub)
  model/reference.py # ref encode + temporal append + re-clamp  (stub)
  model/dit.py       # Wan2.1 DiT reuse notes  (stub)
  utils/weights.py   # torch→MLX conversion  (stub)
tests/{parity,smoke}/
_research/G1_INJECTION.md   # ← the locked spec
refs/Phantom/              # upstream clone (gitignored)
```

## Next gates
- **G3** — pin 1.3B checkpoint repo+commit; confirm `.pth` vs sharded-safetensors load.
- **G4** — eval set: 2 subjects (dialogue) → 4; anime character sheets.
- Then Phase A (substrate-reuse audit) per the handoff.

## License
Apache-2.0. Derived from Phantom-Wan (ByteDance), Wan2.1 (Wan-AI), mlx-video. See `NOTICE`.
