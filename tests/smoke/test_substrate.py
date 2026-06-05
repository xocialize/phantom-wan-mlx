"""Gate A smoke — the reused Wan2.1 substrate loads + runs with Phantom checkpoints.

Parity is inherited from the published Bernini-R / mlx-video substrate; this only
confirms the three components load and produce sane shapes with Phantom's weights.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DIT = ROOT / "weights" / "phantom" / "Phantom-Wan-1.3B.pth"
VAE = ROOT / "weights" / "wan-base" / "Wan2.1_VAE.pth"

pytestmark = pytest.mark.skipif(not DIT.exists() or not VAE.exists(),
                                reason="Phantom + Wan VAE checkpoints required")


def test_dit_loads():
    from phantom_wan_mlx.utils.weights import load_phantom_dit
    model, cfg = load_phantom_dit(DIT)
    assert cfg.dim == 1536 and cfg.num_layers == 30 and cfg.model_type == "t2v"
    from mlx.utils import tree_flatten
    w = dict(tree_flatten(model.parameters()))
    assert float(mx.abs(w["blocks.0.self_attn.q.weight"]).mean()) > 0   # real weights


def test_vae_roundtrip():
    from phantom_wan_mlx.utils.weights import load_wan_vae
    enc = load_wan_vae(VAE, encoder=True)
    dec = load_wan_vae(VAE, encoder=False)
    img = mx.array(np.random.default_rng(0).standard_normal((1, 3, 1, 256, 256)).astype(np.float32))
    z = enc.encode(img)
    assert z.shape == (1, 16, 1, 32, 32)          # 16-ch latent, 8x spatial
    out = dec.decode(z)
    mx.eval(out)
    assert out.shape[1] == 3 and not bool(mx.isnan(out).any())
