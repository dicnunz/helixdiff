# HelixDiff Breakthrough Research: Mac-Local Diffusion LM

Status: live research packet for the 3-hour loop started on 2026-05-21. The Chrome-backed ChatGPT teammate route is still blocked at the Codex Chrome Extension communication layer, after Chrome, the extension, and the native host manifest checked out. I am keeping the GPT-5.5 Pro critique brief here so the second-model review can resume the moment that path is live.

## Claim Boundary

The target cannot honestly be "global SOTA language model on a Mac." Current diffusion-LM frontier work uses large pretraining runs, pretrained AR initializations, broad datasets, or major GPUs. The real opening is narrower and sharper:

**Build the best no-pretrain, no-spend, Mac-local document repair diffusion LM on a fair visible-context-only benchmark.**

This target is not a dodge. It moves the contest to the one place a diffusion LM has native leverage: arbitrary-position repair under frozen context. A static broad model only reads the page; HelixDiff can make that page into an inference-time training set.

## Source Anchors

Primary sources checked during this loop:

- MDLM, arXiv:2406.07524: masked discrete diffusion becomes competitive when the objective and recipe are cleaned up; their abstract says modern masked diffusion reaches a new SOTA among diffusion models and approaches AR perplexity.
- LLaDA, arXiv:2502.09992: diffusion LMs can scale under pretraining/SFT, but the result is an 8B-scale argument, not a laptop-from-scratch route.
- Dream 7B, arXiv:2508.15487: strong open diffusion LLM; the paper attributes the result partly to AR-based LLM initialization and context-adaptive noise rescheduling, which is outside the no-pretrain claim.
- DDOT, arXiv:2506.13579: attacks flexible-position and flexible-length infilling by denoising token values and positions through OT coupling.
- LADD, arXiv:2510.18114: adds auxiliary latent channels to repair factorized reverse transitions and improve few-step cross-token coherence.
- FS-DFM, arXiv:2509.20624: trains step budget explicitly and reports 8-step parity with a 1024-step baseline for long text generation.
- FLDD, arXiv:2605.18204: learns the noising distribution itself; useful because HelixDiff's current corruption policy is still hand-designed.
- Prompt-infilling dLMs, arXiv:2604.03677: shows infilling can be unlocked by changing masking practice, not architecture alone.
- TTARAG, arXiv:2601.11443: updates model parameters during inference to specialize to retrieved passages.
- QueST, arXiv:2605.13369: derives supervision from the input query itself for parameter-efficient test-time self-training.
- TTT layers, arXiv:2407.04620: frames hidden state as a model updated by self-supervised learning on test sequences.
- Analog Bits, arXiv:2208.04202: shows discrete data can benefit from alternate representations and self-conditioning.

## Breakthrough Thesis

The best shot is not a bigger tiny model. It is **ephemeral specialization**.

Name: **DocForge Diffusion**

Mechanism: before solving the withheld gap, compile the surrounding document into temporary weights and constraints. The model makes synthetic holes from allowed text, performs tiny test-time updates, learns a local verifier, then samples only inside the blank while never changing exposed bytes.

The inversion:

```text
normal LM: train once, infer everywhere
DocForge: train broadly enough to denoise, then train again on the exact visible document before answering
```

On Mac hardware, this is the asymmetry. We cannot buy scale. We can buy locality with seconds of adaptation.

## Novelty Matrix

| Existing idea | What it solves | DocForge difference |
| --- | --- | --- |
| MDLM | better masked objective and sampling recipe | use MDLM-style discipline, but make the test document an adaptation substrate |
| LLaDA | diffusion LMs scale as LLMs | avoid broad scaling fight; compete on document-local repair |
| Dream 7B | strong diffusion LLM with AR initialization | no AR checkpoint, no world-knowledge claim, only local structure |
| DDOT | unknown length/position infill | add later as a length lattice; not the first bottleneck |
| LADD | cross-token dependency during few-step denoising | emulate some latent benefit cheaply with a document-local verifier and structure heads |
| FS-DFM | few-step generation stability | train/adapt with the exact sampling budget used on the Mac |
| FLDD | learned forward noising | mine HelixDiff failures into a trainable corruption sampler |
| Prompt infilling | masking practice unlocks infill | turn full-context masking into the central benchmark contract |
| TTARAG | test-time adaptation for RAG | remove retrieval and make visible bytes the adaptation corpus |
| QueST / TTT | input-derived supervision and learned test-sequence state | specialize only on visible bytes, then solve excluded holes with diffusion repair |
| Analog Bits | alternate discrete representation and self-conditioning | add byte-class/lattice channels when exact bytes are too sparse |

I have not found this exact composition in the source map: **visible-document-only test-time adaptation plus verifier-guided remasking for no-pretrain diffusion repair.** That is the defensible novelty claim. "No one has thought of it" is not provable; "this specific mechanism is not in the checked frontier papers and is falsifiable in this repo" is.

## Strongest First Move

Build **Suture TTA** before another checkpoint run.

Why this first:

- It attacks the current failure directly: the 30k checkpoint lost to n-gram bridge on widened hidden spans.
- It can be added without paid compute.
- It creates a clean before/after gate: frozen model versus same model after visible-context adaptation.
- It either produces the Mac-local breakthrough signal or kills the hypothesis in one bench pass.

Concrete shape:

1. Copy the model into a short-lived inference session.
2. Freeze most parameters; train only a tiny adapter, final block, or verifier head.
3. Build synthetic holes only from visible context outside the real hidden target.
4. Run 20-200 micro-steps on those holes.
5. Repair the withheld gap.
6. Delete the temporary weights after the case.

The implementation can start without a perfect LoRA layer: a session-local copy of the final transformer block or output head is enough to prove whether document-local learning helps.

## Benchmark Contract

Name: **DocSutureBench**

Each case is one document with one or more hidden spans. The runner sees all visible bytes and must reconstruct hidden bytes. The masked answer is never included in adaptation examples. Visible context preservation must be exact.

Suites:

- prose: public-domain text and plays;
- code: local source files with identifiers, indentation, brackets;
- markdown: headings, links, tables, bullet rhythm;
- dialogue: speaker labels and turn structure;
- multi-hole: two hidden spans that constrain each other.

Baselines:

- unigram;
- n-gram bridge;
- suffix-array nearest visible span;
- static HelixDiff;
- static HelixDiff with bridge guidance;
- session-adapted denoiser;
- session-adapted denoiser with bridge proposals;
- session-adapted denoiser with verifier remask.

Allowed claim only if:

- no imported pretrained weights;
- no paid API calls in the released runner;
- train and adapt locally on Mac;
- held-out bytes excluded from adaptation;
- visible context is the only test-time training source;
- adaptation without bridge guidance beats n-gram bridge;
- verifier-guided adaptation beats both its raw adapted sampler and the bridge;
- frozen context is 100%;
- results include exact match, byte accuracy, structure accuracy, and failure breakdown.

## 2026-05-21 Proof Update

Suture TTA is now implemented, measurable, and stricter than the first sketch:

- visible adaptation trains a temporary session copy only;
- synthetic adaptation spans exclude the hidden target byte sequence even when the same text appears elsewhere in visible context;
- JSON reports `hidden_target_seen_in_visible_context` separately from `hidden_target_excluded_from_synthetic_targets`;
- frozen visible context remains preserved in tests and benchmark rows.

The new nearest-visible baseline changed the research state. On the 4-case unseen Tiny Shakespeare slice, it got `50.0%` byte accuracy and `50.0%` exact match by copying visible local sutures. Bridge-only got `6.25%`. Static model, bridge-guided model, raw Suture TTA, and bridge-guided Suture TTA all got `0.0%`.

The first retrieval-lattice probe exposed a sharper split. Pure diffusion scoring found the exact answer in the lattice for two cases but rejected those exact candidates. Adding a fixed boundary-suture prior selected the visible exact candidates and matched nearest-visible at `50.0%`. It still did not beat nearest-visible, because the other two holes did not have the answer in the visible candidate lattice.

Next patch landed and got a cheap proof without loading the model: `helixdiff-bench --candidate-oracle-only` now measures whether the repair lattice contains the hidden span before diffusion scoring spends CPU. On the same 4-case unseen validation slice that previously exposed missing candidates, `proof/lattice_oracle_4case.json` reports `oracle_exact_rate: 1.0`. The exact hits are `Gabr` from a name-stem prior, `p--d` and `y-ca` from dash/morpheme bridges, and `lor:` from speaker-label completion.

The follow-up selector audit makes the bottleneck sharper. A fixed structural prior ranks the exact candidate in the top-4 for `4/4` cases and top-8 for `4/4`, with average exact rank `1.5`, but only selects it top-1 in `1/4`. The benchmark now exposes `--lattice-prior-rerank-top-k 4`, which scores only that compact prior set with the diffusion verifier, plus `--lattice-verifier-mode dual`, which averages leave-one-out suture probes with full-hole reconstruction probes. It also separates verifier top-k from sampler top-k; use `--lattice-verifier-top-k 0` so scoring does not erase candidate bytes before measuring them. A one-case smoke proved the dual scores are finite but also exposed a raw verifier failure: diffusion preferred `Nath` over exact `Gabr`, so `--lattice-selector-margin` records whether a calibrated prior anchor blocked that weak override. The retrieval-lattice summary now reports oracle-in-scored-set, prior-anchor exact, raw-verifier exact, anchor exact, anchor-margin gaps, selector-effect counts, outcome-category counts, margin-activation rates, and a no-extra-compute selector-margin sweep. So candidate generation is no longer the first local blocker on this slice. The next breakthrough attempt should spend compute on a calibrated or learned top-k verifier/reranker, not on growing a wider candidate soup.

That does not prove accuracy. It proves the bottleneck moved. Before this patch, the system could not select missing answers. After this patch, the next hard problem is selecting/verifying the right candidate without laundering a hand-built lattice into a model-quality claim.

Source refresh, 2026-05-21: D3PM showed that the discrete corruption transition is a first-class design choice, including absorbing states that connect diffusion to masked and autoregressive modeling (`https://arxiv.org/abs/2107.03006`). SEDD showed that ratio/score-style objectives can make discrete diffusion language models competitive and useful for controllable infilling (`https://arxiv.org/abs/2310.16834`). XDLM's 2026 claim is a useful warning shot: the frontier is moving toward kernels that bridge masked-diffusion understanding and uniform-noise generation, not just toward bigger left-to-right samplers (`https://arxiv.org/abs/2602.01362`). For HelixDiff on a Mac, the actionable translation is narrow: do not chase a huge general LM; build a tiny repair specialist whose corruption kernel, candidate lattice, and verifier calibration are all optimized for visible-context document repair.

That kills the naive version of the breakthrough. Raw visible-context adaptation is not enough. The promising mutation is:

**Retrieval-Lattice Diffusion**: generate a lattice of allowed local repair candidates from visible context, training split bridge guesses, surface-unit splices, byte-class/morphology completions, and sampled diffusion proposals; then use the diffusion model as a verifier/remask controller instead of asking it to invent every byte from scratch.

The next falsifiable edge is no longer generic "sampling." It is a two-stage scoreboard: if the correct answer is in the candidate lattice and the selector misses it, train or calibrate the verifier; if the correct answer is absent, improve candidate generation with byte-class, morphology, and document-structure channels. This turns each failure into a named bottleneck instead of a vague "train bigger" answer.

## Falsifiers

The idea is fake if any of these happen:

- adaptation improves easy synthetic holes but not real held-out holes;
- n-gram bridge still beats the adapted sampler;
- bridge-guided sampling just copies bridge mistakes;
- synthetic adaptation accidentally sees the hidden answer;
- context preservation drops below 100%;
- improvements appear only on Tiny Shakespeare and vanish on code/markdown;
- candidate generation contains good answers, but selection cannot find them;
- latency is so high that the Mac-local story becomes theatrical;
- the benchmark rewards memorized local repetition instead of real repair.

## Implementation Landing Zone

Existing repo hooks already point at the right place:

- `helixdiff/infill.py` parses one `[[hole]]`, preserves frozen context, denoises the span, and ranks candidates with leave-one-out repair scoring.
- `helixdiff/bench.py` creates non-leaky held-out infill cases and compares guide-only, raw-model, and guided-denoiser modes.
- The current proof boundary is known: the 30k suture checkpoint is real but lost the widened gate.

Patch sequence:

1. Add `helixdiff/adapt.py` with a visible-only synthetic-hole generator and short micro-training loop.
2. Add `--adapt-visible-steps` to the infill/bench path.
3. Emit proof fields: adaptation corpus hash, hidden-hole exclusion check, elapsed adaptation time, static score, adapted score, bridge score, frozen-context flag.

First proof target:

```text
static_model_only < bridge_only
session_tta_raw > ngram_bridge
session_tta_verifier > session_tta_raw
frozen_context_unchanged = true for every case
hidden_target_seen_by_adapter = false
```

## GPT-5.5 Pro Handoff Packet

```text
Role:
You are a hostile research partner. Kill weak ideas. Keep only mechanisms that could produce a real no-spend Mac-local edge.

Goal:
Find a breakthrough path for HelixDiff, a from-scratch byte-level masked diffusion LM, to become SOTA in a narrow honest benchmark. Not global LLM SOTA.

Live repo context:
HelixDiff has a 439,968-param 30k Tiny Shakespeare checkpoint, clock/mode conditioning, suture corruption, non-leaky held-out repair bench, bridge baseline, candidate ranking, and slim checkpoint. The widened 8-case gate failed: bridge-only 22.5%, model-only 18.75%, model+bridge 20.0%.

Current candidate:
DocForge Diffusion. Before repairing a held-out gap, train temporary adapters/verifier weights on synthetic holes made only from the visible document. Then repair the withheld bytes with verifier-guided remasking and frozen visible context.

Constraints:
- no paid compute;
- Mac-local;
- no pretrained weights in final model;
- no API model calls in the shipped runner;
- honest public claim;
- mechanism must be falsifiable in the repo.

Checked sources:
MDLM, LLaDA, Dream 7B, DDOT, LADD, FS-DFM, FLDD, prompt-infilling dLMs, TTARAG, QueST, TTT layers, Analog Bits.

What I need:
1. Red-team novelty versus those sources.
2. Pick the one first mechanism to implement.
3. Define the smallest benchmark where a SOTA claim would be honest.
4. Name the failure that would most likely fool us.
5. Give a 3-step HelixDiff implementation plan.

Output:
Under 700 words. Ranked moves only. No encouragement. Include the public claim boundary.
```

## Current Call

Suture TTA shipped and did not clear the stronger gate. Retrieval-lattice selection now matches nearest-visible but does not beat it. The candidate oracle now proves the exact answer enters a reranker-sized top-4 set on the 4-case slice, while the fixed structural prior only picks top-1 once. Do not spend the next loop pretending more raw micro-steps or wider candidate generation are the breakthrough. Build the verifier/reranker next:

1. train or calibrate a tiny diffusion verifier on synthetic visible-document holes;
2. rerank only the structural-prior top-4 candidates with `--lattice-prior-rerank-top-k 4 --lattice-verifier-mode dual --lattice-verifier-top-k 0 --lattice-selector-margin 3.0`;
3. sweep selector margins with `--lattice-selector-margin-sweep` from the same scored candidates so margin calibration is measured rather than guessed;
4. run `helixdiff-calibrate-selector` on the resulting benchmark JSON to pick the lowest safe observed margin frontier without pretending calibration is held-out proof;
5. keep `--lattice-local-prior-calibration` diagnostic-only until a separate proof shows the visible-context weight proposal does not reduce top-4 oracle coverage;
6. report oracle-in-lattice, prior top-k coverage, raw-verifier accuracy, anchor accuracy, margin activation, selected accuracy, and failure category per case.

The first local-prior proof is useful because it failed safely. It records per-case visible-context weight proposals in `proof/lattice_oracle_4case_local_calibration.json`, keeps `applied=false`, and preserves the original top-4 oracle rate at `100%`. This prevents a seductive Mac-local trick from silently making the reranker set worse; the breakthrough standard is not novelty, it is novelty that survives the oracle.

Only call DocForge impressive after verifier-guided lattice selection beats nearest-visible and bridge-only on widened held-out spans. The public line stays severe: **Mac-local SOTA for visible-context document repair, not a general language model.**

## 2026-05-21 Executable Breakthrough Plan

The source refresh has been pinned into a command so the next run does not depend on remembering this memo:

```sh
helixdiff-breakthrough-plan
```

It emits the ranked lanes, source IDs, claim boundary, GPT teammate blocker, and exact proof commands. The current top lane is `strict_repair_lattice_proof`: run the predeclared `helixdiff-proof-recipe` benchmark under the shared Heavy Slot, then run its calibrate and `helixdiff-gate --require-repair-proof-contract` commands. That recipe now requires the strict anchor sweep `prior,surface,visible_reranker` and diagnostic-only visible-reranker calibration, so the next heavy run cannot ignore the newest selector mechanism. The cheap non-heavy next lane is `visible_hole_reranker`: run `helixdiff-calibrate-selector`, inspect its `visible_hole_reranker` receipt, then test the visible-context top-k reranker anchor before spending compute on heavier model adaptation.

The calibrator now makes that lane falsifiable before another training run. On the checked-in one-case smoke, `proof/selector_margin_calibration_smoke.json` reports `status=diagnostic_only_insufficient_cases`, `bottleneck=visible_hole_reranker`, `oracle_in_scored_set_rate=1.0`, `raw_verifier_exact_rate=0.0`, and `lowest_safe_observed_rescue_margin=3.0`. The benchmark now exposes `--lattice-selector-anchor visible_reranker`, `--lattice-visible-reranker-calibration`, and `--lattice-apply-visible-reranker-calibration`, which hide only visible-context spans, sweep tiny prior/surface reranker weights, and reuse the existing selector-margin machinery on the same scored candidates. The strict proof gate now also requires `visible_reranker_selected_exact_rate`, `visible_reranker_top4_exact_rate`, visible-reranker harm/help counts, and per-case visible-reranker calibration receipts before a narrow repair-lattice claim can pass. The next heavy-slot benchmark should widen the case count; if that pattern holds, compare `prior`, `surface`, and `visible_reranker` anchors before training anything larger. If oracle-in-scored-set collapses, stop training the verifier and fix candidate generation instead.

Source-backed updates from the refresh:

- MDLM (`arxiv:2406.07524`) supports treating objective and sampler discipline as real diffusion-LM leverage, not cosmetics.
- Block Diffusion (`arxiv:2503.09573`) supports locality, block structure, and data-driven schedules as the path around random-mask weakness.
- SEDD (`arxiv:2310.16834`) supports discrete score/ratio objectives and controllable infilling as a serious alternative to plain masked CE.
- LLaDA (`arxiv:2502.09992`) supports scratch diffusion LMs at scale, but also proves the laptop claim boundary: HelixDiff cannot honestly borrow 8B-scale results.
- LADD (`arxiv:2510.18114`) supports the diagnosis that factored reverse transitions need auxiliary joint structure; HelixDiff's cheap version is a surface-signature/reranker channel, not a broad foundation-model claim.
- Frequency-informed MDLM work (`2025.babylm-main.38`) supports spending scarce Mac-local gradient signal on rare/hard boundary bytes instead of uniform corruption alone.
