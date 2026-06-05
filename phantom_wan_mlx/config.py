"""Phantom-Wan S2V config (truth from upstream phantom_wan/configs, G3).

1.3B is the locked v1 oracle. Substrate = Wan2.1 (NOT Wan2.2).

Pinned checkpoints (G3, see _research/G3_CHECKPOINT.md):
- Phantom weights: bytedance-research/Phantom @ 926cb19b8273d3841edcf905ca4ddb57f8e43207
    v1: Phantom-Wan-1.3B.pth  (single .pth, ~5.69 GB, torch.load + strict=False)
    14B: Phantom_Wan_14B-*-of-00006.safetensors + index (later)
- Substrate: Wan-AI/Wan2.1-T2V-1.3B @ 37ec512624d61f7aa208f7ea8140a131f93afc9a
    Wan2.1_VAE.pth · models_t5_umt5-xxl-enc-bf16.pth · google/umt5-xxl/
"""
from dataclasses import dataclass

PHANTOM_REPO = "bytedance-research/Phantom"
PHANTOM_REVISION = "926cb19b8273d3841edcf905ca4ddb57f8e43207"
PHANTOM_1_3B_FILE = "Phantom-Wan-1.3B.pth"          # single .pth → torch.load branch
PHANTOM_14B_BASENAME = "Phantom_Wan_14B"            # sharded safetensors (later)

WAN21_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
WAN21_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
WAN21_VAE_FILE = "Wan2.1_VAE.pth"
WAN21_T5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
WAN21_TOKENIZER = "google/umt5-xxl"


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
