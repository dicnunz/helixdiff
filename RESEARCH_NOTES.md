# Research Notes

## Design Decision

HelixDiff uses absorbing-mask diffusion because it keeps the implementation minimal and honest: the forward process corrupts discrete tokens into a MASK state, while the reverse process predicts the original bytes. This matches the practical thread running from D3PM to recent large language diffusion models.

## Novel Additions

### Span-Shock Corruption

Pure independent masking often leaves enough local context that a small model can win by patching nearby characters. Span-shock corruption masks contiguous regions as a second pressure source. The model must infer missing runs from both sides, which is closer to the promise of a bidirectional diffusion LM.

### Ribbon Suffix Corruption

Free-form language generation is harsher than infilling because the model may begin with only a prompt and a field of masks. Ribbon suffix corruption sometimes hides a suffix after a known prefix, giving the denoiser practice at continuation while keeping the same reverse diffusion interface.

### Entropy-Clock Sampling

Fixed unmasking schedules reveal tokens at the same pace regardless of model confidence. HelixDiff estimates normalized entropy over candidate positions each step. Low-entropy regions reveal earlier; high-entropy regions stay masked and get more reverse-process passes.

### Ribbon Decode

Entropy-clock decode is useful for arbitrary infilling. Ribbon decode is the language-generation counterpart: it reveals positions from left to right, but each prediction still comes from the bidirectional denoiser and can condition on all visible scaffold/context, not only a causal cache.

### Confidence Remasking

Autoregressive decoding freezes a token once emitted. HelixDiff can re-mask low-confidence generated positions during intermediate steps. This is a practical way to let the denoiser repair unstable regions without needing a separate critic.

### Corpus Scaffold Guidance

Blank-page diffusion is brutally underconstrained for a laptop-sized byte model. HelixDiff optionally trains a local n-gram guide from the same user-provided corpus, samples a rough scaffold, masks part of it, and lets the denoising transformer repair the holes. This is not a pretrained crutch; it is a small scratch prior used to make the reverse chain start from a language-shaped field.

## Why Byte-Level

A byte tokenizer is not the most efficient tokenizer for a huge LLM, but it is the cleanest scratch boundary. The repo can be downloaded and trained without hidden tokenizer files, pretrained merges, or external model assets.

## Scaling Hypothesis

The larger configs are meant to test whether entropy-clock reveal scheduling becomes more valuable as the denoiser becomes better calibrated. The sampler exposes `--remask`, `--temperature`, and `--top-k` so this can be measured directly.

For serious scale, the key experiment is not only model size. It is the mix between three pressures: high-noise blank-page diffusion, span infilling, and ribbon continuation. HelixDiff keeps those knobs explicit so a larger run can discover the right schedule instead of baking in one decoding ideology.
