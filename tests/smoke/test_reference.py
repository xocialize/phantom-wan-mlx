"""Phase B smoke — reference encode + temporal-tail assemble/strip (G1 injection)."""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
VAE = ROOT / "weights" / "wan-base" / "Wan2.1_VAE.pth"


def test_preprocess_white_pad():
    from phantom_wan_mlx.model.reference import preprocess_ref
    wide = Image.fromarray(np.zeros((300, 900, 3), dtype=np.uint8))   # black, wide
    p = preprocess_ref(wide, 832, 480)
    assert p.size == (832, 480)
    a = np.asarray(p)
    assert (a[0] == 255).all()        # top band is white padding (fill=255)
    assert (a[a.shape[0] // 2, a.shape[1] // 2] == 0).all()   # center is the (black) image


def test_assemble_strip_roundtrip():
    from phantom_wan_mlx.model.reference import assemble_input, strip_refs
    target = mx.zeros((1, 16, 21, 60, 104))
    refs = mx.ones((1, 16, 2, 60, 104))
    full = assemble_input(target, refs)
    assert full.shape == (1, 16, 23, 60, 104)
    assert strip_refs(full, 2).shape == target.shape


@pytest.mark.skipif(not VAE.exists(), reason="Wan VAE checkpoint required")
def test_encode_references_shape():
    from phantom_wan_mlx.model.reference import encode_references
    from phantom_wan_mlx.utils.weights import load_wan_vae
    enc = load_wan_vae(VAE, encoder=True)
    refs = [Image.fromarray(np.random.default_rng(i).integers(0, 255, (600, 400, 3), dtype=np.uint8))
            for i in range(2)]
    z = encode_references(enc, refs, 832, 480)
    mx.eval(z)
    assert z.shape == (1, 16, 2, 60, 104) and not bool(mx.isnan(z).any())
