"""Phase E — export self-contained MLX Phantom-Wan-1.3B for mlx-community.

Single repo holding both DiT precisions + the shared Wan2.1 substrate (avoids
duplicating the 11 GB umT5):
  transformer-bf16.safetensors      (DiT bf16)
  transformer-4bit.safetensors      (DiT int4, group_size 64, embeds/time/head kept hi-prec)
  vae.safetensors                   (Wan2.1 16-ch VAE, bf16)
  t5_encoder.safetensors            (umT5-XXL, bf16; reused from Bernini-R conversion)
  config.json + README.md
"""
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "Phantom-Wan-1.3B-MLX"
DIST.mkdir(parents=True, exist_ok=True)

from phantom_wan_mlx.model import dit as DIT  # noqa: E402
from phantom_wan_mlx.utils import weights as W  # noqa: E402

PH = ROOT / "weights/phantom/Phantom-Wan-1.3B.pth"
VAE = ROOT / "weights/wan-base/Wan2.1_VAE.pth"


def _bf16(m):
    from mlx.utils import tree_map
    m.update(tree_map(lambda a: a.astype(mx.bfloat16) if a.dtype == mx.float32 else a, m.parameters()))
    mx.eval(m.parameters())
    return m


# fixed inputs for the int4 cosine check
m0, cfg = W.load_phantom_dit(PH)
from phantom_wan_mlx.utils.weights import load_umt5, encode_text  # noqa: E402
t5, tok = load_umt5(cfg); ctx = encode_text(t5, tok, "a subject", cfg.text_len); del t5
lat = mx.array(np.random.default_rng(0).standard_normal((16, 5, 60, 104)).astype(np.float32))
rope, seq = DIT.prepare_grid(m0, 5, 60, 104, cfg.patch_size)


def fwd(m):
    o = DIT.forward(m, lat, mx.array([500.0]), ctx, rope, seq)[0]; mx.eval(o)
    return np.array(o.astype(mx.float32)).ravel().astype(np.float64)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


ref = fwd(m0)

# --- DiT bf16 ---
W.save_native(_bf16(m0), DIST / "transformer-bf16.safetensors")
print("saved transformer-bf16")

# --- DiT int4 (scoped) ---
m4, _ = W.load_phantom_dit(PH)
nn.quantize(m4, group_size=64, bits=4, class_predicate=W.quant_predicate); mx.eval(m4.parameters())
cos4 = cos(ref, fwd(m4))
W.save_native(m4, DIST / "transformer-4bit.safetensors")
print(f"saved transformer-4bit (cosine {cos4:.5f})")

# --- VAE bf16 (encoder + decoder halves both needed; save full) ---
vae = _bf16(W.load_wan_vae(VAE, encoder=False))
W.save_native(vae, DIST / "vae-decoder.safetensors")
vae_e = _bf16(W.load_wan_vae(VAE, encoder=True))
W.save_native(vae_e, DIST / "vae-encoder.safetensors")
print("saved vae encoder+decoder")

# --- umT5 (reuse Bernini conversion, copy as-is) ---
shutil.copy(W.BERNINI_T5, DIST / "t5_encoder.safetensors")
print("copied t5_encoder")

cfg_json = {
    "model": "Phantom-Wan-1.3B", "framework": "mlx", "task": "subject-to-video",
    "dtype": "bfloat16", "dim": cfg.dim, "num_layers": cfg.num_layers, "num_heads": cfg.num_heads,
    "in_dim": cfg.in_dim, "patch_size": list(cfg.patch_size), "cross_attention_dim": cfg.text_dim,
    "vae_z_dim": 16, "vae_stride": [4, 8, 8],
    "quantization_4bit": {"group_size": 64, "bits": 4, "skip": list(W.QUANT_SKIP), "cosine_vs_bf16": round(cos4, 5)},
    "sample": {"shift": 5.0, "steps": 50, "guide_img": 5.0, "guide_text": 7.5, "fps": 16},
}
(DIST / "config.json").write_text(json.dumps(cfg_json, indent=2))
sizes = {f.name: round(f.stat().st_size / 1e9, 2) for f in sorted(DIST.glob("*.safetensors"))}
print("\nexported", DIST.name, "| sizes GB:", sizes)
