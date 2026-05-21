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
- Ribbon suffix corruption for prompt-to-continuation pressure.
- Full-attention Transformer denoiser with RMSNorm and SwiGLU blocks.
- Entropy-clock and ribbon reveal schedules with optional confidence remasking.
- Optional scratch n-gram corpus scaffold for guided local sampling.
- High-order bridge guidance from visible left anchors during infill.
- Suture-trace infill reporter for per-step mask repair previews and confidence/entropy trace data.

## Training Data

The included default corpus is a small original seed corpus in `data/seed_corpus.txt` for smoke training only. The repo also includes `data/tinyshakespeare.txt` for a larger public from-scratch demo. Replace both with a larger licensed text corpus for meaningful capability.

## Limitations

- Tiny laptop checkpoints mostly learn the local style of the seed corpus.
- Scaffolded sampling can look better than unguided blank-page diffusion because it uses an n-gram prior trained from the supplied corpus.
- The included exact infill proof uses the provided seed corpus as a scratch bridge guide; it is a mechanism demo, not evidence of broad semantic understanding.
- Byte-level modeling is universal but less sample-efficient than a mature subword tokenizer.
- Diffusion text generation uses multiple network evaluations per sample.
- Large-scale quality depends on corpus quality, training duration, and compute.
