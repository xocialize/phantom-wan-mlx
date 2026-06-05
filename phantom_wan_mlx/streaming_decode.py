"""Lossless streaming (temporal-chunked) VAE decode — Phase D.5.

Whole-sequence `WanVAE.decode` peaks ~2.3 GB/frame and OOMs past ~49 frames on a 128 GB
M5 (standard 81-frame Wan output is impossible). This decodes the latent ONE temporal
chunk at a time, threading the decoder's CausalConv3d cross-chunk cache exactly like the
already-shipped chunked `encode`, so peak memory is flat in length and the output is
**bit-identical** to `vae.decode(z)`.

Consumer-side extension (no mlx-video fork): mlx-video's `ResidualBlock` already threads
`feat_cache`; the two gaps it leaves — the `upsample3d` Resample (its temporal `time_conv`
ignores the cache) and the top-level chunk loop — are filled here. NCHWD `(B,C,T,H,W)`
port of Lance's `vae_stream.py` (NHWC); Phase-1 temporal only (spatial halo-tile deferred).

Reference: lance-mlx `src/lance_mlx/model/vae_stream.py`; mlx-video `wan_2/vae.py`.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.wan_2.vae import CACHE_T, AttentionBlock, ResidualBlock, Resample

_REP = "Rep"   # sentinel: first chunk through an upsample3d skips temporal doubling


def _conv_cached(conv, x, fc, fi):
    """CausalConv3d with cross-chunk temporal cache (NCHWD; mirrors ResidualBlock)."""
    idx = fi[0]
    cache_x = x[:, :, -CACHE_T:]
    if cache_x.shape[2] < 2 and fc[idx] is not None:
        cache_x = mx.concatenate([fc[idx][:, :, -1:], cache_x], axis=2)
    out = conv(x, cache_x=fc[idx])
    fc[idx] = cache_x
    fi[0] += 1
    return out


def _temporal_interleave(tc, b, c, t, h, w):
    """mlx-video upsample3d channel->frame interleave: [B,2C,T,H,W] -> [B,C,2T,H,W]."""
    tc = tc.reshape(b, 2, c, t, h, w)
    return mx.stack([tc[:, 0], tc[:, 1]], axis=3).reshape(b, c, t * 2, h, w)


def _spatial_upsample(rs, x):
    """Per-frame nearest-2x + Conv2d (identical to stock; no temporal mixing)."""
    b, c, t, h, w = x.shape
    xf = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)   # [BT,H,W,C]
    xf = mx.repeat(xf, 2, axis=1)
    xf = mx.repeat(xf, 2, axis=2)
    xf = rs.resample[1](xf)                                   # Conv2d -> [BT,2H,2W,C//2]
    co = xf.shape[-1]
    return xf.reshape(b, t, h * 2, w * 2, co).transpose(0, 4, 1, 2, 3)


def _resample_upsample3d_cached(rs, x, fc, fi):
    """Streaming upsample3d. mlx-video's stock upsample3d ALWAYS doubles every frame (no
    first-chunk frame-0 skip, unlike diffusers/Wan22) — so this is just a cached `time_conv`
    (causal zero-pad on the first chunk, prev-chunk tail thereafter) + interleave + spatial.
    """
    b, c, t, h, w = x.shape
    idx = fi[0]
    cache_x = x[:, :, -CACHE_T:]
    if cache_x.shape[2] < 2 and fc[idx] is not None:
        cache_x = mx.concatenate([fc[idx][:, :, -1:], cache_x], axis=2)
    tc = rs.time_conv(x, cache_x=fc[idx])                    # fc[idx]=None first chunk -> zero-pad
    fc[idx] = cache_x
    fi[0] += 1
    x = _temporal_interleave(tc, b, c, t, h, w)
    return _spatial_upsample(rs, x)


def _decoder_chunk(decoder, x, fc, fi):
    """One latent chunk through Decoder3d, threading the shared temporal cache."""
    x = _conv_cached(decoder.conv1, x, fc, fi)
    for layer in decoder.middle:
        x = layer(x) if isinstance(layer, AttentionBlock) else layer(x, feat_cache=fc, feat_idx=fi)
    for layer in decoder.upsamples:
        if isinstance(layer, ResidualBlock):
            x = layer(x, feat_cache=fc, feat_idx=fi)
        elif isinstance(layer, Resample):
            if layer.mode == "upsample3d":
                x = _resample_upsample3d_cached(layer, x, fc, fi)
            else:                                            # upsample2d: per-frame, no temporal
                x = layer(x)
        else:
            x = layer(x)
    x = nn.silu(decoder.head[0](x))                          # head: RMS_norm, silu, CausalConv3d
    x = _conv_cached(decoder.head[2], x, fc, fi)
    return x


def decode_streaming(vae, z, chunk_lat: int = 1):
    """Lossless temporal-chunked decode. Bit-identical to vae.decode(z), flat peak memory.

    z: normalized latent [B, z_dim, T_lat, H, W] -> video [B, 3, T_out, H, W] in [-1,1].
    """
    mean = vae.mean.reshape(1, -1, 1, 1, 1)
    inv_std = vae.inv_std.reshape(1, -1, 1, 1, 1)
    z = z / inv_std + mean

    t_lat = z.shape[2]
    fc = [None] * 64                                         # generous; one slot per cached conv
    outs = []
    for start in range(0, t_lat, chunk_lat):
        zc = z[:, :, start:start + chunk_lat]
        xc = vae.conv2(zc)                                   # kernel-1, per-frame, no cache
        fi = [0]
        oc = _decoder_chunk(vae.decoder, xc, fc, fi)
        # materialize the carried cache too — else fc holds lazy slice-views into this
        # chunk's freed buffers, which alias/go stale across the boundary (fails at >2 chunks)
        mx.eval(oc, *[c for c in fc if isinstance(c, mx.array)])
        outs.append(oc)
    return mx.clip(mx.concatenate(outs, axis=2), -1, 1)
