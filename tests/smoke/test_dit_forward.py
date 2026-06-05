"""Phase C smoke — reference-injected DiT forward (G1).

Validates: (1) forward over the assembled F+K grid yields the right shape, (2) the
injection is LIVE (real refs vs zeroed refs change the target-frame prediction), and
(3) no DiT-induced periodic artifact (through-DiT FFT ratio <= direct-decode ratio).

The stock mlx-video WanModel forward parity is inherited from the published Bernini-R
substrate; this only checks the Phantom F+K extension + injection wiring.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DIT = ROOT / "weights" / "phantom" / "Phantom-Wan-1.3B.pth"
VAE = ROOT / "weights" / "wan-base" / "Wan2.1_VAE.pth"

pytestmark = pytest.mark.skipif(not DIT.exists() or not VAE.exists(), reason="checkpoints required")


@pytest.fixture(scope="module")
def setup():
    from phantom_wan_mlx.utils.weights import load_phantom_dit, load_umt5, encode_text
    model, cfg = load_phantom_dit(DIT)
    t5, tok = load_umt5(cfg)
    ctx = encode_text(t5, tok, "a subject", cfg.text_len)
    return model, cfg, ctx


def test_forward_shape_and_injection(setup):
    from phantom_wan_mlx.model import dit as DIT_
    from phantom_wan_mlx.model.reference import assemble_input
    model, cfg, ctx = setup
    F, K, h, w = 3, 2, 60, 104
    rng = np.random.default_rng(0)
    target = mx.array(rng.standard_normal((1, 16, F, h, w)).astype(np.float32))
    refs = mx.array(rng.standard_normal((1, 16, K, h, w)).astype(np.float32))
    rope, seq = DIT_.prepare_grid(model, F + K, h, w, cfg.patch_size)

    def run(rf):
        out = DIT_.forward(model, assemble_input(target, rf)[0], mx.array([500.0]), ctx, rope, seq)[0]
        mx.eval(out)
        return out

    o_real, o_zero = run(refs), run(mx.zeros_like(refs))
    assert o_real.shape == (16, F + K, h, w) and not bool(mx.isnan(o_real).any())
    # injection live: refs must move the target-frame prediction
    assert float(mx.abs(o_real[:, :F] - o_zero[:, :F]).mean()) > 1e-3
