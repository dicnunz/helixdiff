# Research Notes

## Design Decision

HelixDiff uses absorbing-mask diffusion because it keeps the implementation minimal and honest: the forward process corrupts discrete tokens into a MASK state, while the reverse process predicts the original bytes. This matches the practical thread running from D3PM to recent large language diffusion models.

## Novel Additions

### Span-Shock Corruption

Pure independent masking often leaves enough local context that a small model can win by patching nearby characters. Span-shock corruption masks contiguous regions as a second pressure source. The model must infer missing runs from both sides, which is closer to the promise of a bidirectional diffusion LM.

### Boundary-Pinned Suture Corruption

The held-out benchmark asks the model to repair one interior hole with visible context on both sides. Earlier training exposed the model to random masks, span shocks, and suffix ribbons, but not enough exact benchmark-shaped repairs. Suture corruption adds a mode where one bounded interior span is masked and the rest of the row stays visible.

This is not a sampler trick. It changes the supervised reverse-process signal so the denoiser sees the same kind of problem the public infill gate asks it to solve.

The suture curriculum can also upweight the first and last masked byte in the hidden span. Those two bytes touch the visible context and are the highest-leverage repair positions; when they are wrong, the span visibly fails to stitch back into the surrounding text even if its interior bytes look plausible.

### Ribbon Suffix Corruption

Free-form language generation is harsher than infilling because the model may begin with only a prompt and a field of masks. Ribbon suffix corruption sometimes hides a suffix after a known prefix, giving the denoiser practice at continuation while keeping the same reverse diffusion interface.

### Clock And Mode Conditioning

A diffusion denoiser should know what kind of corruption it is reversing. HelixDiff now conditions every forward pass on three related signals:

- continuous diffusion time;
- bucketed remaining mask fraction;
- corruption mode: random mask, span shock, ribbon continuation, or infill repair.

The intent is to reduce stage confusion. A model repairing a nearly finished infill span should not behave like a model facing a high-noise blank page.

### Entropy-Clock Sampling

Fixed unmasking schedules reveal tokens at the same pace regardless of model confidence. HelixDiff estimates normalized entropy over candidate positions each step. Low-entropy regions reveal earlier; high-entropy regions stay masked and get more reverse-process passes.

### Ribbon Decode

Entropy-clock decode is useful for arbitrary infilling. Ribbon decode is the language-generation counterpart: it reveals positions from left to right, but each prediction still comes from the bidirectional denoiser and can condition on all visible scaffold/context, not only a causal cache.

### Confidence Remasking

Autoregressive decoding freezes a token once emitted. HelixDiff can re-mask low-confidence generated positions during intermediate steps. This is a practical way to let the denoiser repair unstable regions without needing a separate critic.

### Corpus Scaffold Guidance

Blank-page diffusion is brutally underconstrained for a laptop-sized byte model. HelixDiff optionally trains a local n-gram guide from the same user-provided corpus, samples a rough scaffold, masks part of it, and lets the denoising transformer repair the holes. This is not a pretrained crutch; it is a small scratch prior used to make the reverse chain start from a language-shaped field.

### High-Order Bridge Guidance

The first guide only looked one byte left and one byte right, which made infill too myopic: a span beginning after `The model begins with a ` should be able to use that whole anchor, not just the final space. The guide now stores scratch n-gram transitions and, during denoising, adds the strongest available visible left-context distribution for each masked position. In ribbon mode with `--max-reveal-per-step 1`, each newly repaired byte becomes context for the next one.

This is deliberately not a pretrained language model. It is closer to a small phrase memory built from the same local corpus, used as a bridge prior while the Transformer still runs the reverse diffusion step.

### Suture-Trace Infill

`helixdiff.infill` turns the diffusion LM into a visible repair instrument. A user marks one span as `[[missing text]]`; the CLI replaces that span with mask tokens, freezes every other byte, runs the denoising loop, and writes a trace containing remaining masks, reveal counts, remask counts, entropy, confidence, and a preview string.

The current proof repairs `sentence` exactly from eight mask bytes:

```text
The model begins with a ~~~~~~~~, removes bytes until the page looks damaged.
The model begins with a s~~~~~~~, removes bytes until the page looks damaged.
The model begins with a se~~~~~~, removes bytes until the page looks damaged.
The model begins with a sen~~~~~, removes bytes until the page looks damaged.
The model begins with a sent~~~~, removes bytes until the page looks damaged.
The model begins with a sentence, removes bytes until the page looks damaged.
```

The point is not to overclaim the tiny checkpoint. The point is to expose a diffusion-native behavior that an autoregressive sample cannot show as naturally: anchored text repair with a reversible visible field.

### Self-Suture Candidate Ranking

Sampling one reverse chain from a tiny byte model is noisy. HelixDiff can now run several candidate repair chains for the same frozen context, then score each proposed repair with leave-one-out probes inside the repaired span. Each probe masks one proposed byte while leaving the rest of the candidate and both frozen boundaries visible.

This is stricter than blanking the whole hole again. Whole-hole scoring mostly rewards generic high-frequency completions; leave-one-out scoring asks whether the repaired bytes cohere with each other and with the visible suture edges. The selected repair is not chosen by a separate classifier or hosted model; it is a diffusion-native best-of-k loop.

In the Tiny Shakespeare 12k proof, this change flipped bridge-guided held-out infill from underperforming bridge-only by `-2.5` byte-accuracy points to beating bridge-only by `+5.0` points on the same 4-case unseen-hole candidate suite. The 30k suture-curriculum continuation preserved a narrow 4-case lift, but the widened 8-case suite failed: bridge-only reached `22.5%`, model-only reached `18.75%`, and model+bridge reached `20.0%`.

That failure is now part of the artifact. The scorer is useful enough to expose a repair-shaped behavior, but not reliable enough to support a model-quality claim.

### Non-Leaky Bridge Benchmark

The bridge guide is useful, but it can also fool the eye. A phrase-repair demo can look intelligent when the guide is simply retrieving an in-corpus continuation. `helixdiff.bench` attacks that failure mode:

- build marked infill cases only from a validation tail;
- train the n-gram bridge only on the training prefix;
- compare unigram baseline, bridge-only baseline, unguided diffusion, and bridge-guided diffusion;
- report whether target holes appeared in the guide training split;
- record checkpoint and split SHA-256 hashes;
- label the checkpoint with a quality tier instead of letting samples speak alone.

The current 30k Tiny Shakespeare checkpoint is rated `mechanism_checkpoint`. Its narrow 4-case repair lift does not survive the 8-case gate, so the evidence does not yet support a strong model-quality claim. That is a feature of the benchmark, not a failure to hide: it makes the next training run honest.

### Claim Gate

The project now treats model-quality language as an executable boundary. `helixdiff.gate` reads a baseline benchmark and a candidate benchmark, then checks:

- masked CE improved;
- masked accuracy rose by at least `1.5%` absolute;
- unguided model-only infill beat bridge-only;
- bridge-guided model infill beat bridge-only;
- frozen context stayed unchanged.

If any check fails, the allowed public claim is mechanism-only. The 30k suture checkpoint fails the widened gate on masked accuracy and bridge lift, so its public boundary is mechanism-only. This is directly aimed at the most dangerous small-model failure mode: samples can look better because the scratch guide helped, while the neural denoiser itself did not add new held-out signal.

### Slim Checkpoint Export

Training checkpoints keep optimizer and EMA state for continuation. Public checkpoints should be smaller and simpler. `helixdiff.export` writes the selected EMA or raw state into `model_state`, drops optimizer state, preserves scratch metadata and mechanism names, then reloads the exported artifact as a smoke check.

## Why Byte-Level

A byte tokenizer is not the most efficient tokenizer for a huge LLM, but it is the cleanest scratch boundary. The repo can be downloaded and trained without hidden tokenizer files, pretrained merges, or external model assets.

## Scaling Hypothesis

The larger configs are meant to test whether entropy-clock reveal scheduling becomes more valuable as the denoiser becomes better calibrated. The sampler exposes `--remask`, `--temperature`, and `--top-k` so this can be measured directly.

For serious scale, the key experiment is not only model size. It is the mix between three pressures: high-noise blank-page diffusion, span infilling, and ribbon continuation. HelixDiff keeps those knobs explicit so a larger run can discover the right schedule instead of baking in one decoding ideology.

## Mac-SOTA Track

The realistic Mac-local target is not global SOTA. It is the strongest replayable from-scratch diffusion LM artifact that can be trained and audited on commodity Apple hardware.

`configs/mac_sota.json` is the current target:

- sequence length `256`;
- width `256`;
- `8` Transformer layers;
- `8` attention heads;
- about `8.8M` parameters;
- span/ribbon corruption mixed throughout training;
- boundary-pinned suture corruption for infill-shaped supervision;
- clock/mode conditioning for corruption-stage awareness;
- EMA checkpointing and resume support in the trainer.

The acceptance gate is intentionally mechanical:

1. train from scratch on a licensed corpus;
2. save raw and EMA weights with checkpoint metadata;
3. run `helixdiff-verify-scratch`;
4. run `helixdiff-bench` on held-out spans;
5. run `helixdiff-gate` against the previous checkpoint;
6. export a slim loadable checkpoint with `helixdiff-export`;
7. publish the exact JSON proof, command, checkpoint hash, and split settings.
