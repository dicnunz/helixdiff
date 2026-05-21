# HelixDiff Model Card

## Model

HelixDiff is a byte-level masked diffusion language model. It predicts masked bytes with a bidirectional Transformer and generates text by iteratively unmasking positions.

## Scratch Boundary

- No pretrained weights.
- No hosted model API calls.
- No Hugging Face `transformers` dependency.
- Tokenizer is implemented in `helixdiff/tokenizer.py`.
- Training writes checkpoint metadata with `scratch_only: true`.

## Intended Use

- Research and education around diffusion language modeling.
- Small-corpus experiments.
- Infilling and iterative text generation experiments.
- Visible span-repair demos where one marked hole is denoised while surrounding bytes stay frozen.
- A clean base for larger from-scratch pretraining.

## Not Intended For

- Production advice, legal/medical/financial decisions, or safety-critical generation.
- Claims that the included tiny checkpoint is comparable to commercial LLMs.
- Training on unlicensed private corpora.

## Architecture

- UTF-8 byte vocabulary with PAD, MASK, BOS, and EOS.
- Absorbing mask forward process.
- Span-shock corruption for contiguous denoising pressure.
- Boundary-pinned suture corruption for held-out span repair pressure.
- Optional suture-boundary loss weighting for repair-specialized continuation runs.
- Ribbon suffix corruption for prompt-to-continuation pressure.
- Continuous time plus bucketed mask-fraction and corruption-mode conditioning.
- Full-attention Transformer denoiser with RMSNorm and SwiGLU blocks.
- Entropy-clock and ribbon reveal schedules with optional confidence remasking.
- Optional scratch n-gram corpus scaffold for guided local sampling.
- High-order bridge guidance from visible left anchors during infill.
- Suture-trace infill reporter for per-step mask repair previews and confidence/entropy trace data.
- Self-suture candidate ranking: multiple reverse chains can be rescored with leave-one-out probes inside the repaired span, so the denoiser judges both boundary fit and internal repair coherence.
- Non-leaky held-out benchmark harness comparing unigram, bridge-only, unguided diffusion, and bridge-guided diffusion on validation-only spans.
- Claim gate that compares a candidate checkpoint against a baseline checkpoint and refuses strong-model language unless the denoiser beats both the previous model and the scratch bridge-only baseline.
- Proof reports include checkpoint and split SHA-256 hashes for replayability.
- Slim checkpoint exporter that turns resumable training checkpoints into smaller EMA-only artifacts for public download.

## Training Data

The included default corpus is a small original seed corpus in `data/seed_corpus.txt` for smoke training only. The repo also includes `data/tinyshakespeare.txt` for a larger public from-scratch demo. Replace both with a larger licensed text corpus for meaningful capability.

## Current Checkpoint Evidence

The checked-in 30k Tiny Shakespeare checkpoint is a mechanism checkpoint:

- `439,968` parameters.
- `30,000` scratch-family training steps on Apple MPS, resumed only from an earlier scratch HelixDiff checkpoint.
- Slim public artifact: `checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_slim.pt`.
- Slim artifact SHA-256: `3d1ae0b04275291f44c17660eeef12c627ed0d8f96132eba3b8caff27bedd9bf`.
- Scratch verifier: `proof/scratch_verifier_clock_suture_30k.json`, `ok: true`.
- Eval smoke: `proof/eval_clock_suture_30k.json`, masked loss `3.327`, masked accuracy `14.73%`.
- Narrow 4-case unseen repair bench: bridge-only `15.0%`, model-only `20.0%`, model+bridge `20.0%`.
- Wider 8-case unseen repair bench: bridge-only `22.5%`, model-only `18.75%`, model+bridge `20.0%`.

That means the current public checkpoint demonstrates the training/sampling/infill/verification stack and a narrow repair signal, but the signal does not survive the widened held-out gate. Model-quality claims should be made only after `helixdiff-bench` shows robust lift over the bridge-only baseline.

The active improvement track is `configs/mac_sota.json` and larger licensed-corpus training with boundary-pinned suture corruption plus clock/mode conditioning. A new checkpoint should not replace the public claim boundary unless `helixdiff-gate` passes against the old Tiny Shakespeare benchmark on a widened held-out suite.

## Limitations

- Tiny laptop checkpoints mostly learn the local style of the seed corpus.
- Scaffolded sampling can look better than unguided blank-page diffusion because it uses an n-gram prior trained from the supplied corpus.
- The included exact infill proof and 30k repair checkpoint are mechanism demos, not evidence of broad semantic understanding.
- The benchmark intentionally separates validation spans from the guide training split; this can make demos look less flattering, but it prevents leakage.
- Byte-level modeling is universal but less sample-efficient than a mature subword tokenizer.
- Diffusion text generation uses multiple network evaluations per sample.
- Large-scale quality depends on corpus quality, training duration, and compute.
