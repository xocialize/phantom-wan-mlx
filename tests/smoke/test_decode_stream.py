"""Phase D.5 — lossless streaming VAE decode: bit-identity + flat memory.

Correctness is gated on **mx.cpu** (true fp32): streaming must be BIT-EXACT to whole-seq
`vae.decode`. On the Apple GPU, fp32 conv is tf32-like (~1e-2), so chunked vs whole-seq
differ by reduction-order noise there — both are equally valid, neither more correct (the
MuseTalk/Zonos lesson). So we assert bit-exactness on cpu, sane output on gpu.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
VAE = ROOT / "weights" / "wan-base" / "Wan2.1_VAE.pth"

pytestmark = pytest.mark.skipif(not VAE.exists(), reason="Wan VAE checkpoint required")


@pytest.mark.parametrize("t_lat", [1, 2, 3, 5])
def test_streaming_bit_exact_cpu(t_lat):
    mx.set_default_device(mx.cpu)
    from phantom_wan_mlx.streaming_decode import decode_streaming
    from phantom_wan_mlx.utils.weights import load_wan_vae

    vae = load_wan_vae(VAE, encoder=False)
    z = mx.array(np.random.default_rng(t_lat).standard_normal((1, 16, t_lat, 16, 16)).astype(np.float32))
    ref = vae.decode(z)
    strm = decode_streaming(vae, z, chunk_lat=1)
    mx.eval(ref, strm)
    assert strm.shape == ref.shape
    assert float(mx.abs(strm - ref).max()) == 0.0   # bit-exact in true fp32


def test_streaming_flat_memory_gpu():
    """Peak memory must be ~flat in length (the whole point); long videos must not OOM."""
    mx.set_default_device(mx.gpu)
    from phantom_wan_mlx.streaming_decode import decode_streaming
    from phantom_wan_mlx.utils.weights import load_wan_vae

    vae = load_wan_vae(VAE, encoder=False)
    peaks = {}
    for t_lat in (5, 21):                                # 17 vs 81 output frames
        z = mx.array(np.random.default_rng(0).standard_normal((1, 16, t_lat, 60, 104)).astype(np.float32))
        mx.eval(z)
        mx.reset_peak_memory()
        v = decode_streaming(vae, z, chunk_lat=1)
        mx.eval(v)
        peaks[t_lat] = mx.get_peak_memory() / 1e9
    # 81-frame peak within ~15% of 17-frame peak => flat in length
    assert peaks[21] < peaks[5] * 1.15, f"memory not flat: {peaks}"
