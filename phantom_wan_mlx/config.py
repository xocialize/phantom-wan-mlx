"""Phantom-Wan S2V config (truth from upstream phantom_wan/configs, G3).

1.3B is the locked v1 oracle. Substrate = Wan2.1 (NOT Wan2.2).
"""
from dataclasses import dataclass


@dataclass
class PhantomWanConfig:
    # transformer (wan_s2v_1_3B.py)
    dim: int = 1536
    ffn_dim: int = 8960
    freq_dim: int = 256
    num_heads: int = 12
    num_layers: int = 30
    in_dim: int = 16
    out_dim: int = 16
    text_dim: int = 4096
    patch_size: tuple = (1, 2, 2)
    window_size: tuple = (-1, -1)
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    model_type: str = "t2v"          # text-only cross-attn; NO CLIP image encoder
    # vae / text (shared_config.py + wan_s2v_1_3B.py)
    vae_z_dim: int = 16
    vae_stride: tuple = (4, 8, 8)
    text_len: int = 512
    num_train_timesteps: int = 1000
    sample_fps: int = 16
    # inference defaults (generate.py / subject2video.generate)
    shift: float = 5.0
    sample_steps: int = 50
    guide_scale_img: float = 5.0
    guide_scale_text: float = 7.5

    @staticmethod
    def s2v_1_3b() -> "PhantomWanConfig":
        return PhantomWanConfig()

    @staticmethod
    def s2v_14b() -> "PhantomWanConfig":
        return PhantomWanConfig(dim=5120, ffn_dim=13824, num_heads=40, num_layers=40)
