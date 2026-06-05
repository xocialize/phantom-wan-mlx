"""Weight loading for Phantom-Wan — consumes the mlx-video Wan2.1 substrate.

Phantom-Wan-1.3B is a finetune of the stock Wan2.1-T2V-1.3B WanModel, so its .pth
loads through mlx-video's Wan sanitizer unchanged (825/826 keys; the 1 gap is the
computed `freqs` RoPE buffer, set at WanModel init).

Substrate weights (reused, no re-port):
  - DiT:  weights/phantom/Phantom-Wan-1.3B.pth      -> mlx-video WanModel
  - VAE:  weights/wan-base/Wan2.1_VAE.pth           -> mlx-video WanVAE (z_dim=16)
  - umT5: reuse Bernini-R's converted t5_encoder.safetensors (same umt5-xxl, mlx-video fmt)
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_video.models.wan_2 import convert
from mlx_video.models.wan_2.config import WanModelConfig
from mlx_video.models.wan_2.wan_2 import WanModel

# reused umT5 (Wan2.2's umt5-xxl == Wan2.1's; mlx-video text_encoder format)
BERNINI_T5 = Path("/Users/dustinnielson/DEV_INT/bernini-r-mlx-weights/ckpt-bf16/t5_encoder.safetensors")


def load_phantom_dit(pth_path: str | Path, variant: str = "1.3B"):
    """Load the Phantom-Wan DiT into the stock mlx-video WanModel. Returns (model, cfg)."""
    cfg = WanModelConfig.wan21_t2v_1_3b() if variant == "1.3B" else WanModelConfig.wan21_t2v_14b()
    model = WanModel(cfg)
    assert "freqs" in dict(tree_flatten(model.parameters())), "WanModel must init `freqs` buffer"

    raw = convert.load_torch_weights(str(pth_path))
    san = convert.sanitize_wan_transformer_weights(raw)
    model.load_weights(list(san.items()), strict=False)   # strict=False: `freqs` stays as init
    mx.eval(model.parameters())
    return model, cfg


def load_wan_vae(pth_path: str | Path):
    """Load the Wan2.1 16-ch VAE (mlx-video WanVAE)."""
    from mlx_video.models.wan_2.vae import WanVAE

    vae = WanVAE(z_dim=16)
    raw = convert.load_torch_weights(str(pth_path))
    san = convert.sanitize_wan_vae_weights(raw) if hasattr(convert, "sanitize_wan_vae_weights") else raw
    vae.load_weights(list(san.items()), strict=False)
    mx.eval(vae.parameters())
    return vae
