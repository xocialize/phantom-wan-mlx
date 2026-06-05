---
license: apache-2.0
library_name: mlx
tags:
- mlx
- text-to-video
- subject-to-video
- video-generation
- phantom
- wan
base_model: bytedance-research/Phantom
pipeline_tag: text-to-video
---

# Phantom-Wan-1.3B — MLX

Apple-MLX port of **[Phantom-Wan-1.3B](https://github.com/Phantom-video/Phantom)** (ByteDance, ICCV 2025) —
**multi-subject subject-to-video (S2V)**: compose up to 4 distinct characters from reference images into
one coherent generated video. Runs natively on Apple Silicon. Apache-2.0.

Phantom-Wan-1.3B is a fine-tune of the stock **Wan2.1-T2V-1.3B** DiT, so this port rides the
[`mlx-video`](https://github.com/Blaizzy/mlx-video) Wan2.1 substrate; the only net-new surface is the
reference-injection path (refs encoded as trailing temporal latent frames + dual-scale chained CFG).

## Contents (self-contained)

| File | What |
|------|------|
| `transformer-bf16.safetensors` | DiT, bf16 (2.84 GB) |
| `transformer-4bit.safetensors` | DiT, int4 group-size 64 (0.98 GB) — embeds/time/head kept hi-precision, cosine 0.99633 vs bf16 |
| `vae-encoder.safetensors` / `vae-decoder.safetensors` | Wan2.1 16-ch VAE (bf16) |
| `t5_encoder.safetensors` | umT5-XXL text encoder (bf16) |
| `config.json` | architecture + sampling defaults |

## Features

- **Multi-subject (≤4 refs)** composition from reference sheets.
- **Lossless streaming VAE decode** — peak memory flat ~20 GB at any video length; 81-frame output decodes fine (whole-sequence decode OOMs past ~49 frames).
- Dual-scale chained CFG (`guide_img=5.0`, `guide_text=7.5`), FlowUniPC scheduler (shift 5, 50 steps).

## Usage

```python
from phantom_wan_mlx import pipeline_mlx as P   # github.com/xocialize/phantom-wan-mlx
P.s2v(
    "two friends walking together in a park",
    ["subjectA.png", "subjectB.png"],
    "out.mp4",
    size=(832, 480), frame_num=81,
)
```

## Parity

DiT loads turnkey into the mlx-video `WanModel` (825/826 keys; the gap is the computed RoPE buffer);
forward + VAE numerics inherited from the published mlx-video / Bernini-R Wan2.1 substrate. Streaming
decode is **bit-exact** to whole-sequence decode on `mx.cpu`.

## License

Apache-2.0 (mirrors upstream Phantom + Wan2.1). Port by MVS Collective (xocialize-code).
