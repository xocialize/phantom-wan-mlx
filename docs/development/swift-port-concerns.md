# Phantom-Wan-1.3B — MLX-Swift port: pre-flight concern map

**Date:** 2026-06-05 · **Status:** Python port complete + published; Swift port not started.
**Purpose:** identify what's de-risked vs. what needs net-new Swift work *before* committing to the port,
with the umT5 text path investigated in depth (and a compensating escape hatch already wired).

## TL;DR

The Python port's headline win — "substrate-reuse, ~160-LOC delta over mlx-video's Wan2.1" — **does not
transfer to Swift**, because there is no Swift `mlx-video`. A Swift port re-stands-up the substrate. The
good news: **`longcat-avatar-mlx-swift` already ships Swift `UMT5EncoderModel` + `AutoencoderKLWan` (both
parity-tested) + a working swift-transformers umT5 tokenizer path** — so the text encoder and VAE are
reuse, not re-port. The genuinely net-new Swift work is the **stock Wan2.1 DiT**, the **FlowUniPC
scheduler**, the **streaming decode**, and the **~160-LOC reference-injection delta**.

## What's DE-RISKED (Swift precedent exists in `longcat-avatar-mlx-swift`)

| Component | Swift status | Note |
|---|---|---|
| **umT5 encoder model** | `Sources/.../Models/UMT5EncoderModel.swift` (parity-tested) | Same `google/umt5-xxl` — reuse LongCat's Swift model **and its converted weights** (Phantom uses the identical umT5; no re-convert). ⚠ key format is diffusers-style (`gate_proj`), *not* Phantom's published mlx-video `ffn.fc1/fc2` — so load LongCat's umT5 weights, not Phantom's `t5_encoder.safetensors`. |
| **umT5 tokenizer** | swift-transformers `Tokenizers` (T5 SentencePiece), used by `LongCatAvatarPipeline.swift` | Works for English in LongCat. **Chinese parity UNTESTED** (LongCat has no tokenizer-parity test, no CJK coverage) — see the umT5 deep-dive below. |
| **Wan VAE** | `Sources/.../Models/AutoencoderKLWan.swift` (parity-tested) | ⚠ LongCat's VAE is the **diffusers-schema** 16-ch (keep-and-project at the dim_mult boundary), architecturally *different* from mlx-video's stock 16-ch `wan_2.vae` that Phantom decodes with. Confirm it reproduces Phantom's decode, or port the stock VAE. |
| **q4/q8 quantized load** | LongCat Swift S3.6.q (`WeightLoader.applyDiTQuantization`) | The skip-pattern quant-load pattern is established; adapt the predicate to Phantom's `quant_predicate` skip-set. |

## What's NET-NEW Swift work (no precedent)

1. **Stock Wan2.1 DiT.** LongCat's Swift `LongCatVideoDiT.swift` is the *custom* single-stream/BSA 13.6B
   model — **not** the stock `WanModel` (WanAttentionBlock: self-attn + t2v cross-attn + 6×dim AdaLN
   modulation) Phantom rides. This is the largest lift. No known stock-Wan Swift DiT to reuse.
2. **FlowUniPC scheduler** — multistep predictor-corrector; net-new Swift.
3. **Streaming VAE decode** — port `streaming_decode.py` (CausalConv3d feat_cache; the `upsample3d`
   always-doubles caveat) to Swift. Straightforward but net-new. Gate bit-identity on the CPU device.
4. **Reference-injection delta** (~160 LOC) — white-pad preprocess, temporal-tail assemble + per-step
   re-clamp + strip, dual-scale chained CFG. Small, mechanical.
5. **Weight export.** The published `mlx-community/Phantom-Wan-1.3B` is **mlx-video native format**; a
   Swift DiT (fresh port) needs its own key layout. Plan a Swift-targeted re-export (or a Swift-side
   key remap).

## umT5 deep-dive — the flagged concern

The model and tokenizer have Swift precedent (above). The **real risk is the text-cleaning step**, not the
model. mlx-video's `encode_text` runs `_clean_text` first, which applies **`ftfy.fix_text`** +
double-`html.unescape` + whitespace-collapse. The docstring flags it: *"critical for correct tokenization
of the Chinese negative prompt."* **`ftfy` has no Swift equivalent**, and Phantom's prompts are Chinese.

**Measured behavior (2026-06-05):**
- `ftfy.fix_text` **does change** Phantom's Chinese prompts: fullwidth CJK punctuation `，` (U+FF0C) →
  ASCII `,`, etc. English prompts: unchanged. Skipping it → wrong tokens → wrong text conditioning.
- **`unicodedata.NFKC` reproduces `ftfy` exactly for the real prompts** (neg + Chinese pos + English).
  Swift has NFKC natively: `str.precomposedStringWithCompatibilityMapping` — **no dependency**.
- NFKC and ftfy diverge only on rare typography: ftfy *uncurls* quotes (`“”`→`"`) which NFKC doesn't, and
  NFKC *expands* `…`→`...` which ftfy doesn't. Generation prompts almost never contain these.

**Swift cleaner recipe (matches ftfy for normal prompts):**
```
NFKC(text)
  → replace “ ” ‘ ’ with " " ' '          (uncurl quotes; ftfy does, NFKC doesn't)
  → html-unescape twice
  → collapse \s+ to " ", trim
```
Then tokenize with the umt5 SentencePiece (swift-transformers), **append EOS id 1, no BOS**, pad/truncate
to `text_len=512`.

**Parity gate:** `goldens/umt5_tokenizer_golden.json` holds `{raw, cleaned, token_ids}` for English +
Chinese + an edge case. When the Swift tokenizer lands, assert its `clean→tokenize` output equals these
ids. A mismatch on a normal prompt = a cleaner/tokenizer bug; a mismatch only on the ellipsis edge case =
known + acceptable.

## Compensating dev already in place (this session)

- **Escape hatch:** `pipeline_mlx.s2v(precomputed_context=…, precomputed_context_null=…)` bypasses the
  cleaner + tokenizer + 11 GB umT5 **entirely** — and `pipeline_mlx.encode_prompt(prompt)` produces the
  `[L,4096]` context offline. So a Swift demo can ship Python-encoded prompt embeddings and defer the
  whole umT5 text path indefinitely, or use it as a fallback for any prompt the Swift tokenizer mis-handles.
- **Tokenizer goldens** captured (above) as the Swift parity gate.

## Recommended Swift sequencing (when we get there)

1. Reuse LongCat's Swift `UMT5EncoderModel` + tokenizer + `AutoencoderKLWan`; verify the VAE schema
   actually decodes Phantom latents (else port stock `wan_2.vae`). Run the tokenizer goldens **with a
   Chinese prompt first** — that's the one untested thing.
2. Port the stock Wan2.1 DiT (the real work) + FlowUniPC + streaming decode.
3. Wire the ~160-LOC injection delta. Ship behind the precomputed-context escape hatch first, then enable
   the live tokenizer once its Chinese goldens pass.
