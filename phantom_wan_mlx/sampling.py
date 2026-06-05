"""Dual-scale chained CFG sampler for S2V (G1).

Per step, THREE DiT forwards (locked vs subject2video.generate):
  neg    = model(cat[target, ZERO refs],  context=null)
  pos_i  = model(cat[target, refs],       context=null)
  pos_it = model(cat[target, refs],       context=text)
  noise_pred = neg + w_img*(pos_i - neg) + w_text*(pos_it - pos_i)
Defaults: w_img=5.0, w_text=7.5. Subject-"uncond" = ZEROED ref latent (not dropout/absence).
Scheduler: FlowUniPC, shift 5.0, 50 steps. Reuse mlx-video FlowUniPCScheduler.

TODO(P-D): implement loop + per-step ref re-clamp + tail strip.
"""
raise NotImplementedError("S2V dual-scale CFG loop — see _research/G1_INJECTION.md")
