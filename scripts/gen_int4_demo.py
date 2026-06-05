"""Validate the published int4 artifact end-to-end (load native int4 -> generate)."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mlx_video.models.wan_2.config import WanModelConfig
from mlx_video.models.wan_2.wan_2 import WanModel
from mlx_video.models.wan_2.vae import WanVAE
from phantom_wan_mlx.utils.weights import load_native, quant_predicate, load_umt5, encode_text
from phantom_wan_mlx.model.reference import encode_references
from phantom_wan_mlx.sampling import sample_s2v
from phantom_wan_mlx.streaming_decode import decode_streaming
from phantom_wan_mlx.pipeline_mlx import NEG_PROMPT, _save_video

D = "dist/Phantom-Wan-1.3B-MLX"
cfg = WanModelConfig.wan21_t2v_1_3b()
m4 = WanModel(cfg)
nn.quantize(m4, group_size=64, bits=4, class_predicate=quant_predicate)
load_native(m4, f"{D}/transformer-4bit.safetensors")
t5, tok = load_umt5(cfg, f"{D}/t5_encoder.safetensors")
prompt = ("A little girl with twin ponytails in a light green dress crouches by daisies in a sunny "
          "meadow; a fluffy brown-and-white dog sits beside her, tongue out.")
ctx = encode_text(t5, tok, prompt, cfg.text_len)
ctx_null = encode_text(t5, tok, NEG_PROMPT, cfg.text_len)
del t5
enc = WanVAE(z_dim=16, encoder=True); load_native(enc, f"{D}/vae-encoder.safetensors")
refs = [Image.open(f"weights/refs/assets/ref{i}.png") for i in (1, 2)]
ref_lat = encode_references(enc, refs, 832, 480); del enc
x0 = sample_s2v(m4, ref_lat, ctx, ctx_null, cfg, f_latent=5, h_latent=60, w_latent=104,
                steps=25, seed=42, verbose=True)
del m4
dec = WanVAE(z_dim=16); load_native(dec, f"{D}/vae-decoder.safetensors")
video = decode_streaming(dec, x0[None], chunk_lat=1); mx.eval(video)
print("WROTE", _save_video(video, "outputs/phantom_int4_demo.mp4", fps=16), flush=True)
