"""Wan2.1 DiT reuse notes.

Backbone = stock Wan2.1 (same family as LongCat-Video base). Reuse mlx-video's Wan
patch-embed, blocks (self-attn + t2v text cross-attn), 3D RoPE, Head. No new params.
The DiT forward is UNCHANGED vs base Wan2.1 — Phantom only changes the *input assembly*
(reference.py) and the *guidance loop* (../sampling.py). model_type='t2v' (no CLIP).

TODO(P-A): wire to the LongCat Wan2.1 substrate modules; confirm parity.
"""
