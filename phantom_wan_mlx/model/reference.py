"""Reference-injection path — the only net-new surface (G1).

Mechanism (locked vs phantom_wan/subject2video.py + modules/model.py):
  1. Each ref image -> aspect-preserving resize + WHITE pad -> [-1,1] -> VAE.encode
     with a singleton temporal dim -> one latent frame [C,1,Hl,Wl].
  2. Concatenate K ref frames along the TEMPORAL axis at the TAIL of the target latent:
     model_input = cat([noisy_target[:, :-K], ref_latents], axis=temporal)   # dim=1 in torch
  3. NO special positional handling: refs occupy ordinary trailing 3D-RoPE temporal
     positions F..F+K-1 (reuse stock Wan rope_apply over the extended grid). Separation
     is behavioral, not positional — refs are clean + re-clamped each step + stripped at end.
  4. Re-clamp every denoise step (refs overwritten with clean latents before each forward);
     strip the last K frames from x0 after sampling.

TODO(P-B/P-C): implement against the reused mlx-video Wan2.1 patch-embed + rope.
"""
raise NotImplementedError("reference-injection path — see _research/G1_INJECTION.md")
