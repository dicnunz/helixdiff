# HelixDiff

HelixDiff is a scratch-built diffusion language model: byte tokenizer, absorbing corruption, bidirectional Transformer denoiser, sampler, evaluator, tests, and shortcut verifier live in this repo. It does not call OpenAI, Hugging Face model hubs, `transformers`, or pretrained weights.

The included training recipe is deliberately small enough to run on a laptop. The same code exposes larger configs for real pretraining runs.

## Local Proof From This Build

| Check | Result |
| --- | --- |
| Unit tests | `11` tests passing |
| Scratch verifier | no banned pretrained-model snippets; checkpoint says `scratch_only: true` |
| Tiny checkpoint | `433,440` parameters, `2,400` steps on Apple MPS |
| Eval smoke | masked loss `3.105`, masked accuracy `14.6%`, scaffolded sample `189.7` bytes/s |
| Suture-trace infill | repaired `sentence` exactly from `~~~~~~~~`; frozen context unchanged |
| Sample | `HelixDiff learns language as an editable field, generation...` |

Proof files are checked into the repo under `proof/`: `scratch_verifier_tiny.json`, `eval_tiny_final.json`, `sample_tiny_candidate_42.json`, and `infill_demo.json`. The downloadable checkpoint is `checkpoints/helixdiff_tiny.pt`; the human-readable demo text is `samples/latest.txt`.

## What Makes It Different

- **Byte-level absorbing-mask diffusion.** Text is UTF-8 bytes plus four special tokens, so the model can train without a downloaded tokenizer.
- **Span-shock corruption.** Training mixes independent absorbing masks with contiguous spans, forcing the model to learn infilling rather than just local token repair.
- **Ribbon suffix corruption.** Some batches mask a suffix after a visible prefix, so the denoiser can serve both bidirectional infill and language-generation decode.
- **Entropy-clock sampling.** Generation reveals more tokens only when the model is confident; uncertain regions stay masked longer and receive more denoising passes.
- **Ribbon decoding.** For language-shaped continuation, the sampler can reveal masked bytes from left to right while still using the bidirectional denoiser at every step.
- **Confidence remasking.** The sampler can deliberately re-mask low-confidence generated tokens, giving the model a second chance instead of freezing early mistakes.
- **High-order bridge guidance.** The scratch n-gram guide now uses full visible left anchors, not only adjacent bigrams, so a masked span can be sutured from the phrase that leads into it.
- **Suture-trace infill.** A dedicated infill path repairs `[[marked]]` spans while preserving every unmasked context byte and emitting a per-step reveal trace.
- **Corpus scaffold guidance.** An optional scratch n-gram guide can initialize a rough local scaffold, then the diffusion model edits masked holes. No external model or pretrained tokenizer is involved.
- **Scratch-only verifier.** `helixdiff.verify_scratch` scans code and checkpoints for common pretrained-model shortcuts.

## Research Lineage

HelixDiff is not a clone of one paper. It is a compact implementation inspired by the useful parts of:

- D3PM, arXiv `2107.03006`: discrete diffusion and absorbing-state corruption for text.
- SEDD, arXiv `2310.16834`: score-entropy objectives for discrete language generation.
- MDLM, arXiv `2406.07524`: masked diffusion training with efficient token samplers.
- LLaDA, arXiv `2502.09992`: large-scale masked diffusion with a Transformer reverse model.

## Quick Start

```
git clone https://github.com/dicnunz/helixdiff.git
cd helixdiff
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
python -m unittest discover -s tests
```

No `uv` required:

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Train

Tiny laptop proof:

```
helixdiff-train \
  --config configs/tiny.json \
  --steps 600 \
  --checkpoint checkpoints/helixdiff_tiny.pt \
  --sample-out samples/latest.txt
```

Bigger local run:

```
helixdiff-train \
  --config configs/base.json \
  --data data/your_corpus.txt \
  --steps 20000 \
  --checkpoint checkpoints/helixdiff_base.pt
```

Large config skeleton:

```
helixdiff-train \
  --config configs/large.json \
  --data /path/to/large_corpus.txt \
  --steps 250000 \
  --checkpoint checkpoints/helixdiff_large.pt
```

## Sample

```
helixdiff-sample \
  --checkpoint checkpoints/helixdiff_tiny.pt \
  --prompt "In a diffusion language model," \
  --tokens 192 \
  --steps 48 \
  --temperature 0.85 \
  --top-k 48 \
  --remask 0.08
```

Language-shaped continuation mode:

```
helixdiff-sample \
  --checkpoint checkpoints/helixdiff_tiny.pt \
  --guide-data data/seed_corpus.txt \
  --schedule ribbon \
  --scaffold \
  --scaffold-remask 0.05 \
  --prompt "HelixDiff" \
  --tokens 192 \
  --steps 48
```

`--guide-data` trains a local n-gram guide from the text file you provide. `--scaffold` uses it as a scratch-built prior, then masks part of that scaffold for the diffusion transformer to repair. Leave it off for pure blank-page denoising.

## Infill

The most direct demo is span repair. Mark one region with `[[...]]`; HelixDiff replaces only that span with masks, freezes the surrounding context, and runs the same reverse diffusion loop.

```
helixdiff-infill \
  --checkpoint checkpoints/helixdiff_tiny.pt \
  --text "The model begins with a [[sentence]], removes bytes until the page looks damaged." \
  --guide-data data/seed_corpus.txt \
  --guidance 2.0 \
  --schedule ribbon \
  --max-reveal-per-step 1 \
  --json-out proof/infill_demo.json
```

The checked-in proof repairs:

```text
The model begins with a ~~~~~~~~, removes bytes until the page looks damaged.
The model begins with a s~~~~~~~, removes bytes until the page looks damaged.
The model begins with a se~~~~~~, removes bytes until the page looks damaged.
The model begins with a sen~~~~~, removes bytes until the page looks damaged.
The model begins with a sentence, removes bytes until the page looks damaged.
```

## Evaluate

```
helixdiff-eval \
  --checkpoint checkpoints/helixdiff_tiny.pt \
  --data data/seed_corpus.txt \
  --batches 20
```

The evaluator reports masked-token negative log likelihood, masked-token accuracy, and bytes generated per second during a sampler smoke test.

## Verify Scratch Boundary

```
helixdiff-verify-scratch --checkpoint checkpoints/helixdiff_tiny.pt
```

The verifier fails if code imports known pretrained-model surfaces such as `transformers`, `from_pretrained`, `huggingface_hub`, or API model clients. It also checks checkpoint metadata for the `scratch_only` claim written by training.

## Repository Shape

```text
helixdiff/
  tokenizer.py       byte tokenizer
  diffusion.py       absorbing mask corruption and loss
  model.py           full-attention diffusion Transformer
  data.py            byte stream batching
  train.py           scratch training loop
  sample.py          entropy-clock, ribbon, trace, and scaffolded denoising
  infill.py          [[marked span]] repair CLI and JSON proof reporter
  ngram.py           local scratch n-gram and bridge guide for optional sampling scaffolds
  eval.py            masked-token evaluation and sampler smoke test
  verify_scratch.py  shortcut scanner
configs/
  tiny.json
  base.json
  large.json
tests/
  unittest coverage for tokenizer, diffusion, model, sampler, verifier
```

## Honest Model Card

The tiny checkpoint is a proof that the model trains from scratch and generates through iterative diffusion. It is not a foundation model. The impressive part is the complete scratch-built diffusion-LM stack, the novel sampler/corruption mechanics, and the clean path from laptop proof to large pretraining. To make a genuinely strong public model, train `configs/base.json` or `configs/large.json` on a large, licensed corpus with serious compute.

## Included Data

`data/seed_corpus.txt` is a small hand-written seed corpus for laptop smoke tests. `data/tinyshakespeare.txt` is the public Tiny Shakespeare corpus commonly mirrored from Andrej Karpathy's char-rnn example data, included so users can run a larger-from-scratch proof without hunting for a dataset.
