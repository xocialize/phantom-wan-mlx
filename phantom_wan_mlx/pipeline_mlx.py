"""Phantom-Wan S2V inference entry (MLX). Renderer scope: text + reference VAE latents.

    from phantom_wan_mlx import pipeline_mlx as P
    P.s2v(ckpt, "two friends walking", reference_images=["a.png","b.png"], output_path="out.mp4")

reference_images: list of paths (multi-subject ≤4, each a distinct subject). See G1 doc.
"""
from __future__ import annotations
from pathlib import Path


def s2v(model_dir: str | Path, prompt: str, reference_images: list[str], **kwargs) -> str:
    raise NotImplementedError("pipeline — scaffold only; see _research/G1_INJECTION.md")
