# G1 — Reference-injection mechanism (LOCKED)

**Gate:** G1 — Injection mechanism locked. **No MLX code before this is closed.**
**Status:** ✅ **CLOSED 2026-06-05** — read against upstream `Phantom-video/Phantom` @ `main` (HEAD `bd84b60`), files cited below (cloned in `refs/Phantom/`).
**Headline:** Phantom-Wan S2V injects references as **clean trailing temporal frames** with **ordinary 3D-RoPE positions** and separates conditions with **dual-scale chained CFG** — *not* via SA-3D RoPE. This is simpler than the handoff assumed and **diverges from Bernini-R's approach**.

---

## The four G1 questions, answered

### 1. Concat axis
Reference latents are concatenated along the **temporal / frame axis** and appended at the **tail** of the target latent.

- `get_vae_latents` (subject2video.py:159-166): each ref image → `TF.to_tensor` → `(x-0.5)/0.5` (→ [-1,1]) → `vae.encode([img.unsqueeze(1)])` (singleton temporal dim) → one latent frame; then `torch.cat(ref_vae_latents, dim=1)` stacks K refs along **dim=1 (frames)**.
- Target latent length is extended by K: `target_shape = (z_dim, (F-1)//stride[0] + 1 + ref_latents[0].shape[1], H, W)` (subject2video.py:223-225).
- Per denoise step the model input is `torch.cat([latent[:, :-K], ref_latent], dim=1)` (subject2video.py:293) — first F frames = noisy target, last K = clean refs.
- **MLX note:** axis is the latent **temporal** dim. In MLX (mlx-video Wan latents are `[C, T, H, W]`) this is `axis=1`. There is **no sequence-level concat of a separate token block** — it's a plain video-latent extension, then the stock patch-embed flattens it.

### 2. Subject ordering / delimiting
- Multiple subjects = **comma-separated `--ref_image` paths** (generate.py:248-278 `load_ref_images`, `path.split(",")`), each a distinct subject, in list order.
- Each subject → one padded image → **one latent frame**; subjects are just **consecutive trailing frames** in list order. **No delimiter token, no segment id.** Ordering == ref-list order.
- Preprocessing per ref (generate.py:258-276): aspect-preserving LANCZOS resize + **white (255,255,255) center padding** to target size. ⚠️ Differs from the Bernini MLX `_preprocess_ref` (plain BICUBIC stretch) — **match the white-pad exactly** (VAE input normalization is a silent identity killer).

### 3. Reference-token positional treatment  ← the handoff's "highest-risk trap"
**There is none special.** Refs occupy ordinary sequential 3D-RoPE temporal positions `F … F+K-1`.

- `model.forward` patch-embeds the **whole** `[C, F+K, H, W]` latent and builds `grid_sizes` from its shape incl. the ref frames (model.py:524-526); `rope_apply` applies the standard factorized 3D RoPE over that grid (model.py:39-67). Refs are roped as if they were trailing video frames.
- Self-attention is **full** over the whole sequence (no ref mask; `seq_lens` is padding-only) — target ↔ ref attend freely (model.py:127-... `WanSelfAttention`, `flash_attention`).
- **Separation is behavioral, not positional:** refs are (a) clean/un-noised, (b) re-clamped every step, (c) stripped after sampling, and (d) the model is trained to treat the tail frames as references.
- **Consequence for the port:** **no SA-3D RoPE needed.** Reuse stock Wan2.1 `rope_apply` with an extended temporal grid. This is the single biggest divergence from the handoff (which budgeted SA-3D-style positional work) and from Bernini-R (which *does* use SA-3D per-segment phase). It is a **de-risk**, not a new risk.

### 4. CFG composition
**Dual-scale, chained, three forwards per step** (subject2video.py:292-302):

```
neg     = model(cat[target, ZERO_refs], context=null)     # ref_latents_neg = torch.zeros_like(refs)
pos_i   = model(cat[target, refs],      context=null)      # subject only
pos_it  = model(cat[target, refs],      context=text)      # subject + text
noise_pred = neg + guide_scale_img * (pos_i - neg) + guide_scale_text * (pos_it - pos_i)
```

- Defaults: `guide_scale_img = 5.0`, `guide_scale_text = 7.5` (generate.py:214-227; generate() defaults subject2video.py:177-178).
- **Subject "unconditioned" = a ZEROED reference latent** (`torch.zeros_like(ref_latents[0])`, subject2video.py:220), *not* dropout/absence. The ref tail is always present in the sequence; the negative branch just feeds zeros there.
- Plain chained CFG (**not** APG/momentum — contrast Bernini-R r2v which used APG). Scheduler: `FlowUniPCMultistepScheduler`, `shift=5.0`, 50 steps (subject2video.py:269-276).

---

## Per-step re-clamp (implementation-critical)
The scheduler updates **all** F+K latent frames, but each step rebuilds the input as `cat([latent[:, :-K], clean_refs])` so the ref tail is **overwritten with clean refs before every forward** (subject2video.py:293-300). After the loop, `x0 = x0[:, :-K]` strips the refs (subject2video.py:313). The MLX loop must replicate both: re-inject clean refs each step, strip the tail at the end.

## Model / substrate truth (G3 preview)
- `WanModel(model_type='t2v')` → **t2v cross-attn (text only), no CLIP** image encoder (model.py:372-374, 456; generate passes `clip_fea=None`). `in_dim=out_dim=16`, `patch_size=(1,2,2)`, `text_dim=4096`.
- **1.3B (v1 oracle):** dim 1536 · ffn 8960 · heads 12 · layers 30 · qk_norm · cross_attn_norm · eps 1e-6 (`configs/wan_s2v_1_3B.py`). **14B:** dim 5120 · ffn 13824 · heads 40 · layers 40.
- VAE = **Wan2.1_VAE.pth** (z_dim 16, stride (4,8,8)); T5 = umt5-xxl bf16, text_len 512 (`configs/shared_config.py`). **Confirms substrate = Wan2.1**, shared with LongCat-Video — reuse those modules.

## Net delta vs base Wan2.1 (what to actually build)
1. **Reference encode** — white-pad preprocess + VAE-encode K refs to trailing latent frames (`model/reference.py`).
2. **Input assembly + per-step re-clamp + tail strip** (`model/reference.py` + `sampling.py`).
3. **Dual-scale chained CFG loop** with zeroed-ref negative (`sampling.py`).
4. **Weight conversion** for `Phantom_Wan_1.3B` (and 14B) via the existing mlx-video Wan sanitizer (`utils/weights.py`).

Everything else (patch-embed, blocks, 3D RoPE, Head, UMT5, VAE, FlowUniPC) is **reused** from the LongCat Wan2.1 substrate. **No new `mlx-arsenal` primitive. No SA-3D RoPE.**

## Divergences from the handoff (flag for the plan)
- Handoff feared refs "must NOT be read as temporal frames" + SA-3D positional band as the top trap. **Reality:** refs *are* read as trailing temporal frames with normal RoPE → trap dissolves; drop the SA-3D budget.
- Handoff said "reference-dropout CFG." **Reality:** zeroed-latent negative + plain dual-scale chained CFG (no dropout, no APG).
- Bernini-R's `model/multiseg.py` + `rope_sa3d.py` are still a useful *segment-assembly* reference, but Phantom-Wan S2V does **not** replicate that SA-3D path — it's the simpler temporal-append scheme. Don't port Bernini's SA-3D into Phantom.

## Open items rolled to later gates
- **G3:** pin the exact 1.3B checkpoint repo + commit; confirm `.pth` vs sharded-safetensors load path (subject2video.py:110-132 hardcodes base name `Phantom_Wan_14B` for the sharded branch — 1.3B likely loads a single `.pth`).
- **G4:** eval set — start at 2 subjects (dialogue), then 4; anime character sheets.
- Confirm `vae.encode` singleton-temporal handling matches mlx-video Wan VAE encode signature.

## Source citations (refs/Phantom @ bd84b60)
- `phantom_wan/subject2video.py` — get_vae_latents (159-166), target_shape (223-225), neg refs (220), 3-forward CFG (292-302), re-clamp/strip (293, 313), scheduler (269-276).
- `phantom_wan/modules/model.py` — patch-embed + grid_sizes (524-526), rope_apply (39-67), forward (483-569), WanModel defaults (372-387).
- `phantom_wan/configs/wan_s2v_1_3B.py`, `configs/shared_config.py`.
- `generate.py` — multi-subject parse + white-pad (248-278), CFG-scale defaults (214-227).
