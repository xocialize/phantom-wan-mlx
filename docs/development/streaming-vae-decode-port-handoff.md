# Streaming VAE decode — port handoff for Phantom-Wan + Bernini-R

**Date:** 2026-06-05
**Author of analysis:** Claude (session continuation from lance-mlx PR review arc)
**Status:** Pre-port investigation complete; port not yet started.
**Read time:** 5 min. Captures the substrate facts, two surprises uncovered in
the code read, and the recommended sequencing if/when we invest in this work.

## Why this doc exists

Three substantive optimization PRs landed on `xocialize/lance-mlx` in the
past week:

| PR | Description | Lance win |
|---|---|---|
| [#4](https://github.com/xocialize/lance-mlx/pull/4) | DPM-Solver++(2M) scheduler | ~2× wall-clock (30→12 steps) |
| [#6](https://github.com/xocialize/lance-mlx/pull/6) | `memory_mode` tower relay + tiled VAE decode | 32 GB → 16 GB envelope |
| [#7](https://github.com/xocialize/lance-mlx/pull/7) | **Lossless streaming VAE decode** | 12 GB → 8 GB peak decode |

The cross-port question: do any of these transfer to phantom-wan-mlx,
bernini-r-mlx, or longcat-avatar-mlx?

Initial assessment in
[longcat-avatar-mlx/docs/development/lance-mlx-pr-cross-port-analysis.md](../../../longcat-avatar-mlx/docs/development/lance-mlx-pr-cross-port-analysis.md)
identified PR #7 (lossless streaming decode) as the best single-PR win and
made a first cut at effort estimates against LongCat's port. **A
second-pass code read against phantom-wan revealed phantom-wan and bernini-r
share the upstream `mlx_video.models.wan_2.vae` module, which both reshuffles
the priority and reveals two surprises that change the effort estimate.**
This doc captures the corrected picture.

## VAE substrate map across all four ports

| Port | VAE module | Channels | Layout | Streaming-ready? |
|---|---|---|---|---|
| **Lance** | `mlx_video.models.wan_2.vae22.Wan22VAEDecoder` (1150 LOC, vendored) | 48 | NHWC `(B,T,H,W,C)` | ✅ via `lance_mlx.model.vae_stream` (PR #7) |
| **LongCat** | `longcat_video_avatar/models/autoencoder_kl_wan.py` (737 LOC, local port) | 16 | NCHWD `(B,C,T,H,W)` | ⚠️ Blocks ready; orchestrator missing (see Surprise 2) |
| **Bernini-R** | `mlx_video.models.wan_2` (stock) | 16 | NCHWD | ❌ Blocks NOT ready; tiled path is lossy |
| **Phantom-Wan** | `mlx_video.models.wan_2.vae.WanVAE(z_dim=16)` (stock, 629 LOC) | 16 | NCHWD | ❌ Same as Bernini-R |

Lance's algorithm in `vae_stream.py` is the reference; the algorithm
transfers, but the axis-indexing has to be translated from NHWC to NCHWD
for the other three ports.

## Surprise 1 — mlx-video stock is incomplete for streaming

A first read of mlx-video's `wan_2/vae.py` is misleading because the public
API has the right shape:

```python
class ResidualBlock(nn.Module):
    def __call__(self, x, feat_cache=None, feat_idx=None) -> mx.array: ...

class Resample(nn.Module):
    def __call__(self, x, feat_cache=None, feat_idx=None) -> mx.array: ...

class WanVAE(nn.Module):
    def decode(self, z): ...
    def decode_tiled(self, z, tiling_config=None): ...
```

But two pieces are missing under the hood:

### 1a — `Resample.upsample3d` doesn't honor `feat_cache`

`vae.py:280-294` runs `time_conv` unconditionally:

```python
if self.mode == "upsample3d":
    # Temporal upsample via learned conv
    x_t = self.time_conv(x)         # <-- always runs, no first-chunk skip
    x_t = x_t.reshape(b, 2, c, t, h, w)
    x = mx.stack([x_t[:, 0], x_t[:, 1]], axis=3).reshape(b, c, t * 2, h, w)
```

There is no `"Rep"` sentinel pattern (diffusers `WanResample.forward`), no
first-call skip, no cross-chunk cache. The signature accepts
`feat_cache=None, feat_idx=None`, but the upsample3d branch ignores them.

This means a chunked decode through stock mlx-video would either:
- Double the very first frame (wrong; needs first-chunk skip), or
- Bleed temporal context across chunks unmediated (wrong; needs cache).

### 1b — `WanVAE.decode(z)` is whole-sequence only

`vae.py:561-576` runs:

```python
def decode(self, z):
    z = z / inv_std + mean
    x = self.conv2(z)
    out = self.decoder(x)            # <-- whole-sequence; no feat_cache loop
    return mx.clip(out, -1, 1)
```

No `feat_cache = [None] * num_slots`, no chunk loop, no per-chunk reset of
`feat_idx`. The encoder path has this plumbing (`encode` at lines 506-545),
but the decode side never had it wired.

### 1c — `decode_tiled` is LOSSY

`vae.py:578-629` calls into `tiling.decode_with_tiling`, which uses
trapezoidal mask blending across overlapping tiles. That's the same lossy
cross-fade pattern Lance's `Wan22VAEDecoder.decode_tiled` used pre-PR-#7,
and which PR #7 replaced with the lossless halo-tile approach.

**Net:** phantom-wan and bernini-r currently choose between whole-sequence
decode (memory unbounded in frame count) or lossy tiled decode (bit-error
~1-5 px/255 vs whole-seq). Streaming decode is not actually available.

## Surprise 2 — LongCat's port is MORE complete than mlx-video stock

LongCat's `autoencoder_kl_wan.py:Resample.upsample3d` already has the full
"Rep" sentinel pattern (autoencoder_kl_wan.py:270-310):

```python
if self.mode == "upsample3d":
    if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
            feat_cache[idx] = "Rep"        # first call: skip time_conv, mark
            feat_idx[0] += 1
        else:
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < 2:
                if feat_cache[idx] != "Rep":
                    cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
                else:  # feat_cache[idx] == "Rep"
                    cache_x = mx.concatenate([mx.zeros_like(cache_x), cache_x], axis=2)
            if feat_cache[idx] == "Rep":
                x = self.time_conv(x)
            else:
                x = self.time_conv(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
```

This is essentially line-for-line equivalent to Lance's
`vae_stream._resample_upsample3d_cached`. The block API is ready; only the
top-level `AutoencoderKLWan.decode()` doesn't iterate over chunks.

This was a hidden cost saving missed in the first cross-port analysis. The
LongCat port is closer to streaming-ready than estimated.

## Can LongCat un-fork to ride mlx-video stock? (The "flip the equation" question)

**Short answer: no, and that conclusion was already reached during Stage
1.1 of the LongCat port.** See `longcat-avatar-mlx/docs/development/notes/vae-schema-mismatch.md`.

The channel arithmetic at `dim_mult=[1,2,4,4]` boundary differs structurally:

| Stage | diffusers / Meituan | mlx-video stock |
|---|---|---|
| 0 | 384→384 ×3, upsample 384→192 | 384→384 ×3, upsample 384→192 ✓ |
| 1 | **192→384** (conv_shortcut), 384→384 ×3, upsample 384→192 | 192→192 ×3, upsample 192→96 ✗ |
| 2 | 192→192 ×3, upsample 192→96 | 96→96 ×3, upsample 96→48 ✗ |
| 3 | 96→96 ×3 | 48→96, 96→96, 96→96 ✗ |

mlx-video's halve-on-input pattern is architecturally incompatible with
Meituan's keep-and-project pattern — intermediate tensor shapes literally
differ, so it's not reconcilable by key renaming or reshape. Meituan's DiT
was trained against Meituan's VAE distribution, so we also can't swap to a
Wan-AI checkpoint that happens to match mlx-video's expected schema.

**The opposite direction does work and is worth noting**: upstream LongCat's
`autoencoder_kl_wan.py` to mlx-video as a `wan_2/vae_diffusers.py` sibling
class (or as a `schema="diffusers" | "mlx_video"` flag on `WanVAE`). Then:

- LongCat un-forks and rides upstream.
- mlx-video gains the ability to load any diffusers-0.38 / Meituan-pattern
  Wan checkpoint (could matter for future Wan ports).
- One file fewer to maintain in longcat-avatar-mlx.

Effort: ~4-6 hours. Benefit ceiling: single port (LongCat) + future
ports that pick the diffusers schema. **Defer indefinitely** unless a
second project surfaces a need for the diffusers schema.

## Recommended sequencing (revised)

The corrected effort estimates and dependency chain:

### Stage A — Land streaming decode in phantom-wan-mlx as a consumer-side extension

Two beneficiaries (phantom-wan + bernini-r) for one port. **No fork
required.** Mirrors Lance's `vae_stream.py` pattern — Lance is a pure
consumer of `mlx_video.models.wan_2.vae22` and imports the standard
blocks unchanged.

**Why no fork is needed:** Stock `mlx_video.models.wan_2.vae.Resample`
already exposes `self.time_conv` (CausalConv3d) and `self.resample[1]`
(Conv2d spatial) as public attributes, and `CausalConv3d.__call__`
already accepts `cache_x`. So we can implement the cached upsample3d
logic at the orchestrator level by **bypassing** `Resample.__call__` and
calling `rs.time_conv(x, cache_x=...)` + `rs.resample[1](spatial)`
directly. The "Rep" sentinel pattern lives in our free function, not in
mlx-video. (This is exactly what Lance does — see
`vae_stream.py:75-140`.)

**Work breakdown (consumer-side, no mlx-video changes):**

1. **`phantom_wan_mlx/streaming_decode.py`** — mirror Lance's
   `vae_stream.py` structure, translated NHWC→NCHWD and
   `Wan22VAEDecoder`→`Decoder3d`:
   - `_conv_cached(conv, x, fc, fi)` — CausalConv3d cache wrapper
   - `_resample_upsample3d_cached(rs, x, fc, fi, first)` — the "Rep"
     sentinel implementation that calls `rs.time_conv(x, cache_x=...)`
     + applies the spatial nearest-2× + `rs.resample[1]` manually
   - `_residual_block_cached(blk, x, fc, fi)` — straightforward
     (just thread cache through the two CausalConv3d inside)
   - `_decoder3d_chunk(dec, x, fc, fi, first)` — walk
     `dec.upsamples` + `dec.middle` + `dec.head` block-by-block,
     dispatching to the cached versions for the temporal-mixing ops
   - `decode_streaming(vae, z, chunk_lat=1)` — top-level orchestrator:
     allocate `fc = [None] * num_slots`, loop over z chunks along
     axis=2, concatenate output chunks
   - ~250-300 LOC total
2. **`phantom_wan_mlx/tests/test_decode_stream.py`** — mirror Lance's
   bit-identity test: tiny random-init `WanVAE`, assert
   `decode_streaming(vae, z) == vae.decode(z)` bit-exact for T_lat ∈
   {1, 2, 3, 5, 9}. ~250 LOC, weights-free.
3. **Wire into `phantom_wan_mlx/pipeline_mlx.py`** — add
   `lossless_decode: bool = True` kwarg; dispatch
   `decode_streaming(vae, z)` vs `vae.decode(z)`. ~10 LOC.
4. **Spatial halo-tile (optional Phase 2)** — port Lance's
   `_suffix_spatial_tiled` against `Decoder3d`. Lossless replacement for
   `WanVAE.decode_tiled` (which is lossy trapezoidal blend). ~150 LOC.
   Defer until Phase 1 ships and we measure real memory benefit.

**Effort:** Phase 1 = **3-5 hours** including bit-identity gate
(revised down from 4-6 hr because no `Resample` modification is needed).
Phase 2 = +2-3 hours if pursued.

**Upstream path (parallel, optional, fire-and-forget):** refactor the
free functions into a `WanVAE.decode_streaming` method form and submit
a PR to `Blaizzy/mlx-video`. ~30 min refactor on top of the
consumer-side work; same bit-identity test serves both. If accepted,
future phantom-wan / bernini-r versions delete the consumer-side
extension. If not, no impact — we already have it working internally.
Per the user note: upstream maintainer (prince-canuma) is heavily
involved in Gemma ports, so set expectations to zero for fast review.

**Files to touch (internal-only path):**

```
phantom_wan_mlx/streaming_decode.py             (~300 LOC, new)
phantom_wan_mlx/tests/test_decode_stream.py     (~250 LOC, new)
phantom_wan_mlx/pipeline_mlx.py                 (~10 LOC, edit)
```

**Files to touch (additional, optional upstream PR):**

```
(upstream) mlx_video/models/wan_2/vae.py        (~100 LOC, edit; method form of the free functions)
(upstream) mlx_video/models/wan_2/tests/test_decode_stream.py  (~250 LOC, same test)
```

### Stage B — Already covered in Stage A's `pipeline_mlx.py` edit

Under the consumer-side pattern, the pipeline wiring lands as part of
Stage A's ~10 LOC `pipeline_mlx.py` edit:

```python
# phantom_wan_mlx/pipeline_mlx.py
from phantom_wan_mlx.streaming_decode import decode_streaming

def generate(..., lossless_decode: bool = True):
    # ...
    if lossless_decode:
        video = decode_streaming(vae, z, chunk_lat=1)
    else:
        video = vae.decode(z)
```

No separate stage needed.

### Stage C — Mirror into bernini-r-mlx

Two clean options:

**Option C.1 (simplest):** copy `streaming_decode.py` + the test file
into `bernini-r-mlx`. Both repos are ours so this is duplication
without merge-conflict risk. ~30 min including the pipeline wire.

**Option C.2 (cleaner long-term):** factor `streaming_decode.py` into a
tiny shared utility — either as a new module in `mlx-arsenal` (if we
have one — check) or as a thin standalone PyPI / git-submodule package
that phantom-wan and bernini-r both depend on. ~1-2 hours, pays off
the second time we hit this pattern.

Recommend **C.1 first** to validate the pattern works in two consumers,
then refactor to C.2 when/if we add a third consumer.

### Stage D — Port streaming to LongCat's `AutoencoderKLWan` (deferred per user)

Cheaper than the cross-port analysis estimated, because LongCat's
`Resample.upsample3d` already has the `"Rep"` sentinel (Surprise 2). The
missing piece is just the top-level orchestrator: a `decode_streaming`
method on `AutoencoderKLWan` that allocates feat_cache, walks z by chunk,
and threads it through `self.decoder`.

**Effort:** ~2-4 hours including bit-identity gate. (Previously
estimated 4-8 hours.)

Per user direction, **defer LongCat to the end** regardless of effort.

## What this means for the PR #6 `memory_mode` cross-port

Separate from streaming decode, PR #6 (tower relay) is a 1-2 day port for
each beneficiary because each pipeline has different phase boundaries:

| Port | Phases | Effort estimate |
|---|---|---|
| Lance | UND-tower → prefill → GEN-tower → VAE decode (3 phases) | (done in PR #6) |
| LongCat | umT5-XXL → Whisper → DiT → VAE (4 phases) | 1-2 day focused session |
| Bernini-R | umT5 → DiT (Wan2.2-A14B) → VAE (3 phases) | 1 day |
| Phantom-Wan | umT5 → DiT (Wan2.1-1.3B) → VAE (3 phases, smaller scale) | 1 day, but lower priority — model already fits comfortably at 1.3B |

PR #6 is **not** the leverage move for phantom-wan because the 1.3B DiT
isn't memory-constrained. Worth doing for bernini-r and LongCat at their
own pace; not the streaming-decode path.

## What doesn't transfer

- **PR #4 DPM-Solver++(2M)** — phantom-wan uses `FlowUniPCScheduler` (a
  more sophisticated multistep solver than Lance's Euler). PR #4 is
  orthogonal. Skip.
- **Lance's `mx.compile` investigation** — empirically refuted at 6.2B
  params (−2.8% steady-state regression). At phantom-wan's 1.3B scale the
  result might tilt slightly more favorably (dispatch overhead is
  proportionally larger), but the v2 benchmark script is the evidence we
  trust, not speculation. Defer until someone has a specific reason to
  re-measure.

## Reference files

**Algorithm + tests (Lance, in `xocialize/lance-mlx`):**
- `src/lance_mlx/model/vae_stream.py` (426 LOC) — the streaming algorithm
- `tests/test_decode_stream.py` (280 LOC) — the weights-free bit-identity test
- `results/decode_lossless/DECODE_FOOTPRINT_SWEEP.md` — measurement methodology
- `results/decode_lossless/DECODE_MEMORY_FINDINGS.md` — the original "tiled is
  lossy by 1-5 px/255" finding that motivated PR #7

**Resample "Rep" sentinel reference (LongCat):**
- `longcat_video_avatar/models/autoencoder_kl_wan.py:245-340` — diffusers-style
  Resample with `"Rep"` first-call skip. Drop-in copyable for mlx-video stock
  (same NCHWD layout, same primitives).

**Schema mismatch background:**
- `longcat-avatar-mlx/docs/development/notes/vae-schema-mismatch.md` — why
  LongCat can't un-fork onto mlx-video stock as-is

**Existing cross-port draft (now superseded by this doc on the corrections):**
- `longcat-avatar-mlx/docs/development/lance-mlx-pr-cross-port-analysis.md`

## Open questions for the next focused session

1. **Is `decode_streaming` upstreamable to Blaizzy/mlx-video?** Worth opening
   an issue on the repo to gauge receptivity before doing the port; if
   accepted upstream, all downstream consumers benefit without per-project
   vendoring.
2. **Are phantom-wan's `t > N`-frame envelopes memory-bound today?** The
   1.3B DiT runs at 13-18s/step on M5 with 17 frames at 832×480. The
   streaming-decode unlock matters most for *longer* video paths (e.g.
   81-frame Wan2.1-style outputs). Worth measuring `mx.metal.get_peak_memory`
   at 17, 33, 65, 81 frames on the current whole-sequence decode to
   quantify the upside before investing port time.
3. **`chunk_lat=1` vs `chunk_lat=k`?** Lance's PR #7 found chunk_lat=1 was
   the best memory/throughput trade-off at their scale. Phantom-wan may
   have different sweet spot — worth sweeping in the bit-identity test.
4. **Decoder `_count_decoder_cache_slots()` count.** Need to enumerate the
   CausalConv3d instances in `Decoder3d` to size the feat_cache list. ~10 min
   of code inspection during the port itself.

## Measured memory data (2026-06-05) — answers open question #2

Whole-sequence `WanVAE.decode` peak GPU memory, 832×480, on the M5 Max (128 GB), random latent (decode only, no DiT):

| frames | f_latent | decode peak | result |
|--------|----------|-------------|--------|
| 17 | 5 | **47.5 GB** | ok (this was the Phase-D demo length) |
| 33 | 9 | **80.6 GB** | ok |
| 49 | 13 | **115.9 GB** | ok (last comfortable) |
| 65 | 17 | **140.4 GB** | "ok" but **over 128 GB physical → swap** (slow/risky) |
| 81 | 21 | — | **OOM crash** |

**Verdict: open Q#2 is answered — YES, phantom-wan is hard memory-bound on the VAE decode.** Peak grows ~linearly with frame count (~2.3 GB/latent-pixel-frame here): ~49 frames is the last comfortable length (116 GB), 65 frames spills into swap (140 GB), and the standard Wan **81-frame output OOM-crashes**. Whole-sequence decode caps usable output at ~49 frames.

**Correction to this doc's premise:** the existing **lossy `decode_tiled` also CRASHES at 81 frames** — it does *spatial* trapezoidal tiling, but the blow-up is *temporal* (all 21 latent frames decoded at full res simultaneously). So there is **no working long-video decode path today**, lossy or lossless. The streaming/temporal-chunked decode (Stage A) is therefore a **capability unblock for standard-length video, not a quality/memory optimization over a working path.** This upgrades its priority from "nice win" to "required to ship ≥65-frame Phantom output."

**Sequencing implication:** do Stage A (streaming decode) **as the first item of Phase E / a Phase D.5**, before publishing — otherwise the published 1.3B is capped at ≤49 frames (below typical 81-frame Wan output). Effort estimate (3–5 hr consumer-side) stands; benefits bernini-r identically.

## PORTED — Phase 1 complete (2026-06-05)

`phantom_wan_mlx/streaming_decode.py` (`decode_streaming(vae, z, chunk_lat=1)`) + `tests/smoke/test_decode_stream.py`; wired into `pipeline_mlx.s2v(lossless_decode=True)` (default on). 5 tests green.

**Simpler than the doc assumed — no "Rep" sentinel.** mlx-video's stock `upsample3d` **always doubles every frame** (no first-chunk frame-0 skip, unlike diffusers/Wan22/LongCat), so the temporal op is just a cache-threaded `time_conv` (causal zero-pad first chunk, prev-chunk tail after) + the channel→frame interleave + per-frame spatial up. The `_REP` logic from Lance's NHWC version does **not** apply here. ResidualBlock + downsample-Resample already thread `feat_cache`; only the `upsample3d` time_conv and the top-level chunk loop were missing.

**Correctness — BIT-EXACT on `mx.cpu`** (true fp32, T_lat ∈ {1,2,3,5}: max|Δ| = 0.0). On the Apple **GPU** chunked-vs-whole differs ~4e-3 — that is fp32 tf32-like conv reduction-order noise (chunked runs more/smaller convs), NOT a logic bug; whole-seq GPU decode is itself ~tf32 off true-fp32 by the same order. **Gate correctness on cpu** (the MuseTalk/Zonos lesson), not gpu. The debugging arc (5e-3 seed at the first 2-frame ResidualBlock, growing downstream; every sub-component bit-exact in isolation → reduction-order numerics) is the tell.

**Memory — flat in length (the payoff):** streaming GPU peak is **~20 GB at every length** (17f 19.7 / 33f 19.9 / 65f 20.1 / 81f 20.4 / 121f 20.6 GB) and **81- and 121-frame video decode fine** — whole-seq OOM'd at 81f. One-chunk extent (chunk_lat=1).

**⚠️ OOM-ceiling caveat (corrected):** the whole-seq OOM table above (OOM at 81f) was measured **while a SigLIP2 KonIQ training run was consuming unified memory in another process** — so the *absolute* whole-seq cliff sits higher on a clean machine; the per-frame GB figures and the ~linear scaling are still valid, and streaming's flat ~20 GB beats whole-seq's linear growth regardless. Re-measure the clean whole-seq ceiling opportunistically, but it doesn't change the conclusion.

**Phase 2 (spatial halo-tile) still deferred** — Phase 1's flat ~20 GB already unblocks the full 81-frame envelope; the high-res suffix tiling only matters if we push to 4K / much longer or want the floor below ~20 GB.

**Stage C — mirrored to bernini-r ✅ (2026-06-05).** Bernini-R uses the *same* `mlx_video.models.wan_2.vae` 16-ch VAE (confirmed: its encode returns `[1,16,1,…]`), so `streaming_decode.py` dropped in **verbatim** — bit-exact on cpu (T_lat 1/2/3/5, max|Δ|=0.0) against Bernini's own `vae.safetensors`; wired into both decode sites via `_vae_decode(lossless=True)` default; 4 tests. Pushed `xocialize/bernini-r-mlx@495e940`. (LongCat's forked diffusers-schema VAE is the one that would need real adaptation — still deferred.)

## Handoff state

- **No code touched yet.** This doc is the pre-port investigation only.
- **Tasks tracked** in the session task list (TaskCreate IDs 25-28).
- **Phantom-Wan status:** Phase D complete (end-to-end multi-subject S2V),
  unblocked, no regression risk if we add `lossless_decode` as opt-in.
- **Bernini-R status:** Verified to use the same `mlx_video.models.wan_2`
  substrate via its sampling.py imports. Confirmed via grep, not by
  re-reading the entire pipeline.
- **LongCat status:** Deferred per user direction. Effort estimate revised
  downward to 2-4 hours due to Surprise 2.
