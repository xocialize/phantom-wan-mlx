"""Reference-injection path — the only net-new surface (G1).

Mechanism (locked vs phantom_wan/subject2video.py + generate.py):
  1. Each ref image -> aspect-preserving LANCZOS resize + WHITE (255) center pad to the
     video size -> [-1,1] -> VAE.encode with a singleton temporal dim -> one latent frame.
  2. Concatenate K ref frames along the TEMPORAL axis at the TAIL of the target latent:
     model_input = cat([noisy_target, ref_latents], axis=T)   (dim=1 torch C,T,H,W / dim=2 mlx B,C,T,H,W)
  3. NO special positional handling: refs occupy ordinary trailing 3D-RoPE temporal positions
     F..F+K-1 (stock Wan rope over the extended grid). Separation is behavioral.
  4. Re-clamp clean refs every denoise step; strip the last K frames after sampling.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps


def preprocess_ref(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Aspect-preserving LANCZOS resize + white center-pad to (target_w, target_h).

    Byte-for-byte port of generate.py:load_ref_images (fill=(255,255,255)).
    """
    img = img.convert("RGB")
    img_ratio = img.width / img.height
    target_ratio = target_w / target_h
    if img_ratio > target_ratio:          # wider than target -> fit width
        new_w = target_w
        new_h = int(new_w / img_ratio)
    else:                                  # taller -> fit height
        new_h = target_h
        new_w = int(new_h * img_ratio)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    dw, dh = target_w - new_w, target_h - new_h
    padding = (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2)
    return ImageOps.expand(img, padding, fill=(255, 255, 255))


def _to_vae_input(img: Image.Image) -> mx.array:
    """PIL RGB -> [1,3,1,H,W] in [-1,1] (matches TF.to_tensor().sub_(0.5).div_(0.5))."""
    x = np.asarray(img).astype(np.float32) / 255.0     # HWC [0,1]
    x = (x - 0.5) / 0.5                                  # [-1,1]
    x = np.transpose(x, (2, 0, 1))                       # CHW
    return mx.array(x)[None, :, None, :, :]              # [B=1, C=3, T=1, H, W]


def encode_references(vae_encoder, ref_images, target_w: int, target_h: int) -> mx.array:
    """K PIL refs -> trailing latent frames [1, 16, K, h, w] (one latent frame per subject)."""
    latents = []
    for img in ref_images:
        padded = preprocess_ref(img, target_w, target_h)
        z = vae_encoder.encode(_to_vae_input(padded))    # [1,16,1,h,w]
        latents.append(z)
    return mx.concatenate(latents, axis=2)               # cat along temporal axis -> [1,16,K,h,w]


def assemble_input(noisy_target: mx.array, ref_latents: mx.array) -> mx.array:
    """cat([noisy_target, clean_refs], temporal axis). Both [1,16,*,h,w]; refs at the tail."""
    return mx.concatenate([noisy_target, ref_latents], axis=2)


def strip_refs(latent: mx.array, k: int) -> mx.array:
    """Drop the last K reference frames after sampling (x0 = x0[:, :-K] upstream)."""
    return latent[:, :, :-k, :, :]
