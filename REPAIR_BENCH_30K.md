# Repair Bench 30K

This is the public proof note for the 30k Tiny Shakespeare suture-curriculum checkpoint. It includes the good narrow result and the wider failure, because the point of this repo is to make model-quality claims executable instead of aesthetic.

## Checkpoint

| Field | Value |
| --- | --- |
| Full training checkpoint | `checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_ema.pt` |
| Slim public checkpoint | `checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_slim.pt` |
| Slim SHA-256 | `3d1ae0b04275291f44c17660eeef12c627ed0d8f96132eba3b8caff27bedd9bf` |
| Model SHA-256 | `b93ff58e0a8b6651f747119c8fab4fd6c13e2b57092c45b0bac82b2c4b309cd3` |
| Parameters | `439,968` |
| Training | `12k` scratch clock/mode run, resumed to `30k` with suture-heavy repair curriculum |
| Scratch verifier | `proof/scratch_verifier_clock_suture_30k.json`, `ok: true` |

## Results

| Suite | Cases | Guidance | Bridge-only | Model-only | Model+bridge | Lift vs bridge | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `proof/bench_clock_suture_30k_unseen_candidates.json` | 4 | `0.2` | `15.0%` | `20.0%` | `20.0%` | `+5.0` pts | narrow repair signal |
| `proof/bench_clock_suture_30k_unseen_candidates_8case.json` | 8 | `0.2` | `22.5%` | `18.75%` | `20.0%` | `-2.5` pts | wide gate fails |
| `proof/bench_clock_suture_30k_unseen_candidates_8case_g05.json` | 8 | `0.5` | `22.5%` | `18.75%` | `20.0%` | `-2.5` pts | stronger guide does not rescue it |

All suites use validation-only holes with `--require-unseen-hole`; the bridge guide is trained on the training split only. Frozen context preservation is `100%`.

The strict claim gate for the widened 8-case suite is `proof/gate_clock_suture_30k_8case.json`:

```text
masked CE improved: true
masked accuracy gain met: false
model-only beats bridge-only: false
bridge-guided beats bridge-only: false
frozen context preserved: true
claim boundary: mechanism_only_claim_required_do_not_call_model_sota
```

## Commands

```bash
helixdiff-train \
  --config configs/tiny_suture_curriculum.json \
  --data data/tinyshakespeare.txt \
  --steps 30000 \
  --resume checkpoints/helixdiff_tiny_shakespeare_clock_12k_ema.pt \
  --ema-decay 0.995 \
  --checkpoint checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_ema.pt
```

```bash
helixdiff-bench \
  --checkpoint checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_ema.pt \
  --data data/tinyshakespeare.txt \
  --cases 8 \
  --batches 4 \
  --span-chars 10 \
  --guidance 0.2 \
  --temperature 0.45 \
  --top-k 32 \
  --candidates 4 \
  --require-unseen-hole \
  --json-out proof/bench_clock_suture_30k_unseen_candidates_8case.json
```

```bash
helixdiff-gate \
  --baseline proof/bench_shakespeare_4k_unseen_candidates.json \
  --current proof/bench_clock_suture_30k_unseen_candidates_8case.json \
  --json-out proof/gate_clock_suture_30k_8case.json
```

```bash
helixdiff-export \
  --checkpoint checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_ema.pt \
  --out checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_slim.pt \
  --json-out proof/export_clock_suture_30k.json
```

## Claim Boundary

Allowed:

> HelixDiff is a from-scratch, Mac-trained, byte-level masked diffusion language-model research artifact with clock/mode conditioning, suture corruption, visible repair traces, non-leaky held-out benchmarks, and a slim replayable 30k checkpoint.

Not allowed:

- SOTA diffusion LM.
- Strong general language model.
- Evidence of broad generation quality.
- A claim that the 30k learned denoiser reliably beats the scratch n-gram bridge baseline.

The narrow 4-case result is real but did not survive the wider 8-case check. That makes the current checkpoint a mechanism checkpoint, not a model-quality checkpoint.
