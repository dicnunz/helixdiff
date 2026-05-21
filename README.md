# HelixDiff

HelixDiff is a scratch-built diffusion language model: byte tokenizer, absorbing corruption, bidirectional Transformer denoiser, sampler, evaluator, non-leaky benchmark harness, tests, and shortcut verifier live in this repo. It does not call OpenAI, Hugging Face model hubs, `transformers`, or pretrained weights.

The included training recipe is deliberately small enough to run on a laptop. The same code exposes larger configs for real pretraining runs.

## Local Proof From This Build

| Check | Result |
| --- | --- |
| Unit tests | `114` tests passing |
| Scratch verifier | no banned pretrained-model snippets; 30k checkpoint says `scratch_only: true` |
| Latest checkpoint | `439,968` parameters, `30,000` steps on Apple MPS, resumed only from an earlier scratch HelixDiff checkpoint |
| Slim download | `checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_slim.pt`, `1.8 MB`, SHA-256 `3d1ae0b04275291f44c17660eeef12c627ed0d8f96132eba3b8caff27bedd9bf` |
| Eval smoke | masked loss `3.327`, masked accuracy `14.73%`, EMA loaded without migration |
| Latest repair bench | 4 unseen validation gaps with leak-hardened Suture TTA: nearest-visible `50.0%`, retrieval-lattice `50.0%`, bridge-only `6.25%`, static/adapted model paths `0.0%` |
| Latest candidate oracle | 4 same-seed unseen validation gaps: morphology/factor lattice contains the exact hidden span in `4/4` cases; fixed structural prior ranks it top-4 in `4/4` and selects it top-1 in `1/4`; a diagnostic train-split surface verifier selects exact top-1 in `3/4` and keeps exact top-4 in `4/4`; this is reranker proof only, not selected model accuracy |
| Redacted in-document echo oracle | 8 seeded validation gaps: baseline prior lattice contains the hidden span at K=128 in `75.0%`; baseline-preserving prior+redacted-echo union contains it in `100.0%`; echo alone contains it in `75.0%`; shuffled echo containment falls to `15.625%`; target bytes and target anchor window are redacted/excluded before indexing |
| Causal echo-dominance selector | The follow-up selector smoke tries to promote redacted echo candidates only when real anchors beat blank and swapped-edge nulls. It passes as a fail-closed receipt: echo containment stays `100.0%` vs prior `75.0%`, but direct promotion leaves top-4 exact tied at `37.5%` and drops selected exact from `37.5%` to `25.0%`, so the contract is `ready=false`, `apply=false`, and direct selector claims are blocked |
| Wider repair bench | 8 unseen validation holes: bridge-only `22.5%`, model-only `18.75%`, model+bridge `20.0%` |
| Claim gate | failed; `mechanism_only_claim_required_do_not_call_model_sota` |

Proof files are checked into the repo under `proof/`, including `scratch_verifier_clock_suture_30k.json`, `eval_clock_suture_30k.json`, `bench_clock_suture_30k_unseen_candidates_8case.json`, `bench_suture_tta_4case_lastblock10.json`, `lattice_oracle_4case.json`, `lattice_oracle_4case_local_calibration.json`, `lattice_oracle_4case_surface_anchor_calibration.json`, `bench_prior_topk_dual_smoke.json`, `selector_margin_calibration_smoke.json`, `selector_contract_smoke.json`, `visible_reranker_oracle_smoke.json`, `proxy_mask_selector_contract_smoke.json`, `proxy_mask_selector_contract_smoke_contract.json`, `canvas_boundary_smoke.json`, `in_document_echo_oracle_smoke.json`, `echo_dominance_selector_contract_smoke.json`, `echo_dominance_selector_contract_smoke_contract.json`, `gate_clock_suture_30k_8case.json`, `gate_prior_topk_dual_smoke.json`, and `export_clock_suture_30k.json`. The visible proof note is `REPAIR_BENCH_30K.md`.

The current checked-in checkpoint is intentionally not described as a strong language model. The repository is the 10/10 artifact here: a from-scratch diffusion LM stack with novel repair mechanisms, replayable Mac-local training, a slim checkpoint, and a harsh benchmark gate that refuses to launder scaffold memory into model quality.

## What Makes It Different

- **Byte-level absorbing-mask diffusion.** Text is UTF-8 bytes plus four special tokens, so the model can train without a downloaded tokenizer.
- **Span-shock corruption.** Training mixes independent absorbing masks with contiguous spans, forcing the model to learn infilling rather than just local token repair.
- **Boundary-pinned suture corruption.** A training mode masks one bounded interior span while preserving both sides, directly matching the held-out infill benchmark instead of hoping random masks teach repair.
- **Ribbon suffix corruption.** Some batches mask a suffix after a visible prefix, so the denoiser can serve both bidirectional infill and language-generation decode.
- **Clock/mode-conditioned denoiser.** The Transformer sees continuous diffusion time, bucketed remaining-mask fraction, and a corruption mode id for random, span, ribbon, or infill repair.
- **Entropy-clock sampling.** Generation reveals more tokens only when the model is confident; uncertain regions stay masked longer and receive more denoising passes.
- **Ribbon decoding.** For language-shaped continuation, the sampler can reveal masked bytes from left to right while still using the bidirectional denoiser at every step.
- **Confidence remasking.** The sampler can deliberately re-mask low-confidence generated tokens, giving the model a second chance instead of freezing early mistakes.
- **High-order bridge guidance.** The scratch n-gram guide now uses full visible left anchors, not only adjacent bigrams, so a masked span can be sutured from the phrase that leads into it.
- **Suture-trace infill.** A dedicated infill path repairs `[[marked]]` spans while preserving every unmasked context byte and emitting a per-step reveal trace.
- **Self-suture candidate ranking.** Infill can run several reverse chains, then score each repaired span with leave-one-out denoiser probes inside the proposed repair. That lets the model judge internal coherence plus both frozen boundaries instead of rewarding the most generic blank-hole completion.
- **Visible-context Suture TTA.** Optional test-time adaptation copies the checkpoint for one inference session, trains selected weights on synthetic holes made only from visible context, excludes hidden-target byte spans from synthetic targets, reports the visible-context hash and exclusion flag, then deletes the temporary weights.
- **Nearest-visible repair baseline.** The benchmark now includes a suffix-array-style local retrieval adversary that searches only visible bytes on either side of the hole and scores exact left/right boundary sutures without joining across the hidden gap.
- **Retrieval-lattice diffusion scoring.** The benchmark can expose top visible suture candidates plus bridge/unigram proposals, score them with leave-one-out denoiser probes, and select by diffusion score plus boundary, morphology, and surface priors. It reports selected accuracy, oracle coverage, whether a fixed structural prior placed the exact answer in the reranker-sized top-k set, whether the raw verifier, prior/surface anchor, visible reranker anchor, or selector margin actually made the winning call, a per-case outcome category for the next patch, the score gap a margin must clear to keep the anchor, a no-extra-compute selector-margin sweep, and a visible-hole reranker readiness receipt.
- **Train-split bi-anchor gap lattice.** The candidate lattice can now mine exact byte gaps from the training split only: visible left suffix + fixed-length gap + visible right prefix, with backoff anchor sizes. It logs bi-anchor support and source offsets, and the strict proof recipe includes the source as a preregistered candidate generator. Oracle containment from this source is not generation accuracy.
- **Redacted in-document echo lattice.** A model-free candidate source first replaces the hidden target bytes with sentinel bytes, forbids the target and anchor-window region, then mines exact bi-anchor, left/right echo, and morphology echo candidates from the still-visible bytes of the same eval document. The checked-in smoke is transductive visible-document containment evidence only, not model-quality evidence.
- **Causal echo-dominance selector gate.** The selector computes real same-document echo witness evidence against blank and swapped-edge null contexts before any target metric is read. On the checked-in smoke it kills direct echo promotion instead of claiming lift, because containment improved but top-4 selection did not.
- **Train-split repair surface verifier.** The lattice now computes non-leaky surface features from the training split only: whole-word completions, two-sided dash bridges, and speaker-label completions. It can stay diagnostic, or `--lattice-selector-anchor surface` can make this verifier choose the selector anchor that the diffusion score must beat.
- **Visible-context top-k reranker.** `--lattice-visible-reranker-calibration` hides synthetic spans only inside the visible context, sweeps tiny prior/surface reranker weights, annotates candidates with `visible_reranker_rank`, and can route the selector anchor through `--lattice-selector-anchor visible_reranker` or `--lattice-apply-visible-reranker-calibration`. It is a local verifier hypothesis, not a held-out win until predeclared.
- **Visible-context anchor calibration.** `--lattice-local-surface-anchor-calibration` makes synthetic holes only inside already visible context, compares prior-anchor and surface-anchor recovery there, and recommends switching anchors only when surface wins without harming a prior exact hit. It is diagnostic unless `--lattice-apply-local-surface-anchor-calibration` is predeclared for a held-out scored run.
- **Factorized morphology lattice candidates.** The lattice now adds train-corpus word completions, speaker-label completions, name-stem priors, hyphen/dash morpheme bridges, and surface-unit splices, so failures can be separated into "answer absent from lattice" versus "answer present but verifier rejected it."
- **Corpus scaffold guidance.** An optional scratch n-gram guide can initialize a rough local scaffold, then the diffusion model edits masked holes. No external model or pretrained tokenizer is involved.
- **Non-leaky held-out benchmark.** `helixdiff-bench` builds infill cases only from the validation split, trains the bridge guide only on the training split, and compares unigram, bridge-only, nearest-visible, retrieval-lattice, unguided model, bridge-guided model, visible-context-adapted model, and adapted bridge-guided variants.
- **Scratch-only verifier.** `helixdiff.verify_scratch` scans code and checkpoints for common pretrained-model shortcuts.

## Research Lineage

HelixDiff is not a clone of one paper. It is a compact implementation inspired by the useful parts of:

- D3PM, arXiv `2107.03006`: discrete diffusion and absorbing-state corruption for text.
- SEDD, arXiv `2310.16834`: score-entropy objectives for discrete language generation.
- MDLM, arXiv `2406.07524`: masked diffusion training with efficient token samplers.
- LLaDA, arXiv `2502.09992`: large-scale masked diffusion with a Transformer reverse model.
- Dream 7B, arXiv `2508.15487`: open diffusion LLM scale and flexible inference boundary.
- Block Diffusion, arXiv `2503.09573`: blockwise locality, arbitrary-length generation, and the evidence that diffusion LMs need more structure than independent random masking.
- Trainability of MDMs via Blockwise Locality, arXiv `2604.24832`: the warning that random-masking MDMs can be unstable and that locality-aware blockwise models are a plausible path out.
- Latent-Augmented Discrete Diffusion, arXiv `2510.18114`: the few-step generation lesson that factored reverse transitions lose cross-token structure and need auxiliary joint signals.
- CoDD, arXiv `2603.00045`: coupled discrete-diffusion decoding as a warning that independent token repair can be too weak.
- Unlocking Prompt Infilling Capability for dLMs, arXiv `2604.03677`: masking practice itself can unlock infilling, so HelixDiff treats full-context suture masking as a core mechanism rather than a data-loader detail.
- DreamOn, arXiv `2602.01326`: fixed-length masked canvases are a practical infilling blocker. HelixDiff's current public claim stays fixed-span visible-context repair until a variable-length gate exists.
- Scaling Beyond Masked Diffusion Language Models, arXiv `2602.15014`: perplexity alone can mislead across diffusion families, so HelixDiff reports repair accuracy, baselines, and speed-quality tradeoffs.

The honest scale boundary matters: current public diffusion-LM work is measured in billions of parameters and trillions of tokens. HelixDiff is not trying to fake that on a laptop. It is trying to make the strongest small, inspectable, from-scratch diffusion-LM system that a Mac can train and a GitHub reader can reproduce.

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

Mac-SOTA target config, meant for a serious local MPS run rather than the tiny proof checkpoint:

```
helixdiff-train \
  --config configs/mac_sota.json \
  --data data/tinyshakespeare.txt \
  --steps 100000 \
  --ema-decay 0.995 \
  --checkpoint checkpoints/helixdiff_mac_sota.pt
```

`configs/mac_sota.json` is an 8.8M parameter byte diffusion Transformer at sequence length 256. It is designed to be trainable on this Mac class of machine, not to imply global SOTA against large-cluster research systems.

Low-LR continuation from a smaller checkpoint:

```
helixdiff-train \
  --config configs/tiny_continue.json \
  --data data/tinyshakespeare.txt \
  --steps 30000 \
  --resume checkpoints/helixdiff_tiny_shakespeare_clock_12k_ema.pt \
  --ema-decay 0.995 \
  --checkpoint checkpoints/helixdiff_tiny_shakespeare_clock_30k_ema.pt
```

Repair-specialized continuation, used when the benchmark shows real infill lift but weak general masked accuracy:

```
helixdiff-train \
  --config configs/tiny_suture_curriculum.json \
  --data data/tinyshakespeare.txt \
  --steps 30000 \
  --resume checkpoints/helixdiff_tiny_shakespeare_clock_12k_ema.pt \
  --ema-decay 0.995 \
  --checkpoint checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_ema.pt
```

This curriculum raises suture corruption, narrows the general mask range, and doubles loss weight on the two masked bytes touching visible context. The checked-in 30k run showed a narrow 4-case held-out repair lift but failed the wider 8-case gate, so it remains a repair-mechanism checkpoint rather than a model-quality checkpoint.

Resume a run:

```
helixdiff-train \
  --config configs/mac_sota.json \
  --data data/tinyshakespeare.txt \
  --steps 120000 \
  --resume checkpoints/helixdiff_mac_sota.pt \
  --ema-decay 0.995 \
  --checkpoint checkpoints/helixdiff_mac_sota.pt
```

Large config skeleton for external compute:

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
  --candidates 8 \
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

For the experimental Suture TTA path, add visible-context adaptation flags. This does not use the hidden answer as a synthetic target; the JSON proof records both `hidden_target_seen_in_visible_context` and `hidden_target_excluded_from_synthetic_targets`.

```
helixdiff-infill \
  --checkpoint checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_slim.pt \
  --text "The bridge sees both sides: alpha beta [[gamma]] delta alpha beta." \
  --steps 4 \
  --adapt-visible-steps 1 \
  --adapt-batch-size 2 \
  --adapt-train-scope head \
  --json-out proof/infill_suture_tta_smoke.json
```

The current smoke proves the mechanism and reporting path, not a quality win: frozen context remains true, hidden-target byte spans are excluded from visible adaptation targets, and the sample still misses the held-out word.

## Evaluate

```
helixdiff-eval \
  --checkpoint checkpoints/helixdiff_tiny.pt \
  --data data/seed_corpus.txt \
  --batches 20
```

The evaluator reports masked-token negative log likelihood, masked-token accuracy, and bytes generated per second during a sampler smoke test.

## Benchmark

Use this before making any model-quality claim:

```
helixdiff-bench \
  --checkpoint checkpoints/helixdiff_tiny.pt \
  --data data/seed_corpus.txt \
  --cases 4 \
  --batches 6 \
  --json-out proof/bench_seed_tiny.json
```

The same flag family can be used in the non-leaky benchmark:

```
helixdiff-bench \
  --checkpoint checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_slim.pt \
  --data data/tinyshakespeare.txt \
  --cases 1 \
  --batches 1 \
  --require-unseen-hole \
  --adapt-visible-steps 1 \
  --adapt-batch-size 2 \
  --adapt-train-scope head \
  --json-out proof/bench_suture_tta_smoke.json
```

The one-case smoke currently reports no byte-accuracy lift. That is kept as proof discipline: Suture TTA is implemented and measurable, but it has not earned a model-quality claim.

When the Mac is too hot for a full model run, use the candidate-oracle mode. It does not load a checkpoint and does not prove model quality; it only answers whether the repair lattice contains the true hidden span.

```
helixdiff-bench \
  --candidate-oracle-only \
  --data data/tinyshakespeare.txt \
  --cases 4 \
  --span-chars 4 \
  --context-chars 36 \
  --require-unseen-hole \
  --lattice-bi-anchor-candidates 64 \
  --lattice-bi-anchor-sizes 32,24,16,12,8,6,4 \
  --json-out proof/lattice_oracle_4case.json
```

The current proof ladder includes one more cheap proof before the next model-scored run:

```
helixdiff-visible-reranker-oracle-smoke \
  --config configs/proof_visible_reranker_oracle_smoke.yaml \
  --out proof/visible_reranker_oracle_smoke.json \
  --json
```

This smoke is deliberately model-free. It proves the `prior`, `surface`, and `visible_reranker` branches exist, logs visible-reranker calibration with `apply=false`, records leakage flags, and runs shuffle falsification. Its verdict now fails closed unless the shuffle control drops both selected-exact and lattice-coverage rates, and unless blanked/swapped visible-context counterfactuals drop selected-exact while the causal visible-context score does not harm the real cases. The checked-in receipt passes with `model_load=false`, `gold_in_lattice_at_128=1.0` on the 2-case smoke, `visible_reranker.selected_exact=1.0`, zero leakage flags, shuffled selected-exact down at `0.333`, and `counterfactual_context.verdict=pass`. That is proof integrity, not a model-quality claim.

The proxy-mask selector contract smoke freezes a selector before target scoring:

```
helixdiff-proxy-mask-selector-contract-smoke \
  --config configs/proof_proxy_mask_selector_contract_smoke.yaml \
  --out proof/proxy_mask_selector_contract_smoke.json \
  --contract-out proof/proxy_mask_selector_contract_smoke_contract.json \
  --json
```

It chooses among fixed prior/surface/visible-reranker presets using pseudo masks cut only from visible context, holds out part of those proxy masks for a shuffle check, and records `target_metric_used_for_selection=false`. The emitted contract is `ready_for_heldout`, but target rows in the smoke are after-freeze diagnostics, not a selected-generation claim.

If a release note tries to cite the proxy-mask smoke as target-lift evidence, make it pass the stricter claim gate first:

```
helixdiff-proxy-mask-selector-contract-smoke \
  --config configs/proof_proxy_mask_selector_contract_smoke.yaml \
  --require-useful-ratchet \
  --json
```

The checked-in receipt currently has `useful_ratchet=false`, so that claim gate is expected to fail until a later held-out run or stronger proxy selector actually improves the after-freeze target rows. Contract readiness is useful; it is not SOTA evidence.

There is also an experimental `proxy_geometry_mode: target_retrieval` path. It redacts the target bytes, builds the target lattice anyway, and chooses visible-only proxy masks whose candidate-source geometry looks closest to that redacted target lattice before any target metric is computed. That is a sharper selector idea than boundary matching, but the current smoke correctly fails it because pseudo-heldout top-4 does not beat shuffle. It stays a falsified diagnostic lane until that check turns green.

The newer `proxy_geometry_mode: target_shadow` path is stricter: it computes a label-free target lattice fingerprint, including source diversity, selector disagreement, prior entropy, prior top-gap, pairwise candidate edit distance, and duplicate-source pressure, then picks visible-only pseudo masks whose own fingerprints are closest while avoiding the target anchor window. This is the current best proxy-calibration idea from the Extended Pro teammate plus local proof work, but it is not promoted either: the smoke sets `contract_ready=false` until shadow heldout beats shuffle and avoids prior harm.

The follow-up echo-dominance selector tests whether the redacted in-document echo containment signal can become a safe applied selector:

```
helixdiff-echo-dominance-selector-contract-smoke \
  --config configs/proof_echo_dominance_selector_contract_smoke.yaml \
  --out proof/echo_dominance_selector_contract_smoke.json \
  --contract-out proof/echo_dominance_selector_contract_smoke_contract.json \
  --json
```

It computes real same-document echo witness evidence, contrastive blank-anchor evidence, and swapped-edge evidence before any target metrics are read. The checked-in result is intentionally negative: direct echo promotion does not improve top-4 selection, so the receipt passes as `killed_fail_closed`, emits a diagnostic-only contract, and blocks echo-selector target-lift claims.

Current canvas boundary: HelixDiff's claim target is fixed-span visible-context repair. DreamOn-style variable-length infilling is a separate future gate, not something the current benchmark proves.

```
helixdiff-canvas-boundary-smoke \
  --out proof/canvas_boundary_smoke.json \
  --json
```

That receipt passes only for the fixed-span repair plan. A flexible-length or code-infilling claim must fail until this command has a real variable-length gate to detect:

```
helixdiff-canvas-boundary-smoke \
  --require-variable-length-gate \
  --json
```

```
helixdiff-proxy-mask-selector-contract-smoke \
  --config configs/proof_proxy_mask_target_shadow_smoke.yaml \
  --json
```

For the next model-scored run, use `--lattice-prior-rerank-top-k 4 --lattice-verifier-mode dual --lattice-verifier-top-k 0 --lattice-selector-margin 3.0 --lattice-selector-anchor surface --lattice-selector-anchor-sweep prior,surface,visible_reranker --lattice-bi-anchor-candidates 64 --lattice-bi-anchor-sizes 32,24,16,12,8,6,4 --lattice-local-surface-anchor-calibration --lattice-visible-reranker-calibration` to score only the small structural-prior set that the oracle proved contains the answer on this slice while testing whether the non-leaky surface verifier or visible-context reranker is a better selector anchor than raw prior rank. `dual` averages leave-one-out suture scoring with full-hole reconstruction scoring, verifier top-k `0` keeps scoring from masking out candidate bytes that the sampler would not normally pick, and the selector margin prevents a weak diffusion preference from overriding the chosen anchor. The local anchor calibration is a verifier-of-the-verifier: it reports whether same-case visible synthetic holes would have trusted prior, surface, or a prior/surface-weighted visible reranker before the hidden span is scored. The summary now separates raw-verifier exact rate, anchor exact rate, scored-top-k oracle coverage, bi-anchor oracle contribution, surface-verifier and visible-reranker exact/top-k diagnostics, local surface-anchor recommendation, the no-extra-compute `local_surface_anchor_margin_sweep`, margin activation rate, selector effects, outcome categories, anchor-margin gaps, and a counterfactual selector-anchor/margin sweep from the same scored candidates, so the run tells you whether to train the verifier, tune the margin, switch anchors, trust per-case anchor calibration, or widen the lattice instead of hiding all failures inside one accuracy number.

The surface anchor is not yet a model win. The checked-in oracle proof says it is worth testing: `surface_verifier_selected_exact_rate=0.75`, `surface_verifier_top4_exact_rate=1.0`, `surface_verifier_avg_exact_rank=0.5`, and `surface_verifier_harm_count=0` on the 4-case slice. The model-scored widened benchmark still has to prove that this improves selected repair accuracy.

`--lattice-local-prior-calibration` adds a self-supervised diagnostic pass: for each case, HelixDiff hides same-length spans inside the already visible context, sweeps structural prior weights, and records which weights would best recover those known local holes. This does **not** change ranking unless `--lattice-apply-local-prior-calibration` is explicitly set. The checked-in 4-case oracle proof keeps it diagnostic-only because one local proposal would have pushed `y-ca` out of the top-4 rerank set if applied; top-k oracle coverage is more valuable than a clever but unproven local tweak.

The oracle summary reports this as a gate: `local_prior_suggested_top4_delta`, `local_prior_suggested_harm_count`, `local_prior_suggested_help_count`, and `local_prior_applied_rate` make local calibration auditable before it is ever allowed to steer a scored run.

`--lattice-local-surface-anchor-calibration` is the analogous safety check for selector anchors. On the checked-in four-case diagnostic proof, the hidden-span surface verifier still selects the exact top-1 candidate in `3/4` cases, but visible-context anchor calibration recommends staying with `prior` in all four cases because local synthetic holes do not show a conservative surface advantage: `local_surface_anchor_selected_counts={"prior":4}`, visible prior and surface exact rates are both `0.0625`, and `local_surface_anchor_applied_rate=0.0` in `proof/lattice_oracle_4case_surface_anchor_calibration.json`. That turns a tempting surface-anchor idea into a measured hypothesis, not a hand-waved win.

After a scored run, turn the sweep into a calibration receipt:

```
helixdiff-calibrate-selector proof/bench_prior_topk_dual_smoke.json \
  --json-out proof/selector_margin_calibration_smoke.json
```

The calibrator reports exact rate, byte accuracy, rescue/block rates, anchor-gap pressure, the lowest safe margin on the observed frontier, and, when the benchmark includes `selector_anchor_margin_sweep`, the best anchor-plus-margin pair. It also emits `visible_hole_reranker`: a bottleneck receipt that separates candidate-generation failure from raw-verifier failure, counts oracle-in-scored-set and rescue-opportunity cases, records the lowest safe observed rescue margin, and says whether the next cheap move is a visible-context reranker or just more held-out cases. The benchmark side now has that reranker hook: use `--lattice-visible-reranker-calibration --lattice-selector-anchor-sweep prior,surface,visible_reranker` to compare the calibrated visible-context anchor against prior and surface on the same scored candidates. When local surface-anchor calibration is present, it also reports `local_surface_anchor_margins`: the no-extra-compute per-case strategy that follows the anchor chosen from visible synthetic holes and then evaluates each swept margin. Its own claim boundary is strict: a margin, anchor, or reranker chosen from this output is diagnostic until it is predeclared and evaluated on separate held-out cases.

Freeze a calibrated selector into a held-out contract before using it as evidence:

```
helixdiff-selector-contract proof/selector_margin_calibration_strict_repair_8case.json \
  --json-out proof/selector_contract_strict_repair_8case.json \
  --require-ready
```

`helixdiff-selector-contract` records the calibration file hash, source benchmark receipts, chosen anchor, chosen margin, frozen benchmark flags, missing readiness checks, and held-out requirements. `--require-ready` exits non-zero when the case count or recommendation status is still diagnostic, which is the intended behavior for tiny smokes. This makes the selector a predeclared artifact for the next held-out run instead of an in-sample tuning story.

When a contract becomes ready, apply it by loading the JSON instead of retyping the tuned selector settings:

```
helixdiff-bench ... \
  --lattice-selector-contract proof/selector_contract_strict_repair_8case.json \
  --lattice-require-selector-contract-ready \
  --json-out proof/bench_selector_contract_heldout_8case.json
```

The benchmark records the contract path, hash, id, readiness, and frozen anchor/margin in `case_filter.lattice_selector_contract`. The repair proof gate allows reports without a selector contract, but if a contract is present it must be `ready_for_heldout` and hash/id recorded before `--require-repair-proof-contract` can pass.

The checked-in seed-corpus benchmark result is deliberately unforgiving:

| Variant | Held-out span byte accuracy | Exact span match |
| --- | ---: | ---: |
| Unigram baseline | `20.8%` | `0.0%` |
| Bridge-only baseline | `16.7%` | `0.0%` |
| Unguided model | `8.3%` | `0.0%` |
| Bridge-guided model | `16.7%` | `0.0%` |

Masked validation accuracy is `14.9%`. The benchmark labels the tiny seed checkpoint `mechanism_checkpoint`, not `strong_laptop_checkpoint`.

The latest Tiny Shakespeare suture-curriculum runs are harsher and more useful:

| Suite | Bridge-only | Nearest-visible | Retrieval-lattice | Model-only | Suture TTA | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4 held-out validation gaps, `guidance=0.2` | `15.0%` | not yet measured | not yet measured | `20.0%` | not yet measured | narrow old lift |
| 8 unseen validation holes, `guidance=0.2` | `22.5%` | not yet measured | not yet measured | `18.75%` | not yet measured | wide gate fails |
| 8-case repeat with `guidance=0.5` | `22.5%` | not yet measured | not yet measured | `18.75%` | not yet measured | stronger guide does not rescue it |
| 4 unseen validation gaps, leak-hardened Suture TTA, `guidance=0.5` | `6.25%` | `50.0%` | `50.0%` | `0.0%` | `0.0%` | lattice matches retrieval; non-visible holes remain unsolved |

Candidate-oracle coverage on the same 4-case seed is now `100.0%`: `Gabr`, `p--d`, `lor:`, and `y-ca` all enter the lattice through morphology/factor candidates in `proof/lattice_oracle_4case_local_calibration.json`. The new strict 8-case oracle with train-split bi-anchor enabled is harsher: `proof/lattice_oracle_strict_repair_8case.json` reports `oracle_exact_rate=0.5`, `prior_top4_exact_rate=0.5`, `surface_verifier_selected_exact_rate=0.375`, and `bi_anchor_oracle_exact_rate=0.0`. That is not a failure of the proof; it shows the bi-anchor source is wired and measured, but this Tiny Shakespeare slice is still being carried by morphology/surface features. The visible-reranker smoke then checks the selector branch without loading the model: `proof/visible_reranker_oracle_smoke.json` passes with `visible_reranker.apply=false`, so it remains a diagnostic receipt rather than a generation claim.

Benchmark JSON now includes checkpoint SHA-256 plus train/validation split SHA-256 hashes so proof artifacts can be tied to the exact evaluated bytes.

## Verify Scratch Boundary

```
helixdiff-verify-scratch --checkpoint checkpoints/helixdiff_tiny.pt
```

The verifier fails if code imports known pretrained-model surfaces such as `transformers`, `from_pretrained`, `huggingface_hub`, or API model clients. It also checks checkpoint metadata for the `scratch_only` claim written by training.

## Gate Model-Quality Claims

After a stronger checkpoint is trained, compare it against the previous benchmark before changing public claims:

```
helixdiff-gate \
  --baseline proof/bench_shakespeare_4k_unseen_candidates.json \
  --current proof/bench_clock_suture_30k_unseen_candidates_8case.json \
  --json-out proof/gate_clock_suture_30k_8case.json
```

The gate requires lower masked CE, at least `+1.5%` absolute masked accuracy, model-only infill beating the bridge-only baseline, bridge-guided infill beating bridge-only, and frozen context preservation before a model-quality claim is allowed. It separately checks the retrieval-lattice lane: at least four held-out cases by default, frozen context preservation, retrieval-lattice byte accuracy beating bridge-only, and retrieval-lattice byte/exact accuracy beating the nearest-visible repair baseline. If the model gate fails but the retrieval-lattice gate passes, the only allowed upgrade is a narrow repair-lattice claim, not a model-SOTA claim. The checked-in one-case dual smoke now fails this gate on case count even though its lattice result is exact, which is the intended honesty boundary.

For the next widened repair-lattice proof, add `--require-repair-proof-contract` to make the metric gate also require the strict breakthrough receipt: checkpoint and split hashes, unseen-hole filtering, nearest-visible baseline, retrieval-lattice case rows matching the summary, oracle-in-scored-set diagnostics, surface-verifier diagnostics, visible-reranker diagnostics, selector margin and anchor sweeps, local surface-anchor calibration and its no-extra-compute margin sweep, plus visible-context reranker calibration. It also enforces the predeclared flagship recipe: prior rerank top-k `4`, verifier mode `dual`, verifier top-k `0`, selector anchor `surface`, selector margin `3.0`, anchor sweep `prior,surface,visible_reranker`, margin sweep `0,1,2,3,5`, diagnostic-only local surface-anchor calibration, and diagnostic-only visible-reranker calibration. The recipe now emits a selector-contract JSON after calibration so the next held-out selector settings are frozen before they are tested. This keeps a narrow repair win from becoming a vague model-quality claim and makes missing diagnostics or softer benchmark settings fail visibly before the README language changes.

To print the exact no-model oracle, heavy benchmark, calibration, selector-contract, and gate commands for that recipe:

```
helixdiff-proof-recipe
```

To print the source-backed Mac-local breakthrough lanes that justify that recipe and keep the GPT/browser boundary honest:

```
helixdiff-breakthrough-plan
```

## Export A Slim Checkpoint

Training checkpoints include optimizer and EMA state so runs can resume. For GitHub/download use, export the selected weights into a smaller loadable artifact:

```
helixdiff-export \
  --checkpoint checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_ema.pt \
  --out checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_slim.pt \
  --json-out proof/export_clock_suture_30k.json
```

By default the exporter writes EMA weights as `model_state`, drops optimizer state, verifies the exported checkpoint can load, and preserves the scratch-only metadata.

## Repository Shape

```text
helixdiff/
  tokenizer.py       byte tokenizer
  diffusion.py       absorbing mask corruption and loss
  model.py           full-attention diffusion Transformer
  data.py            byte stream batching
  train.py           scratch training loop
  sample.py          entropy-clock, ribbon, trace, and scaffolded denoising
  adapt.py           visible-context test-time adaptation for repair sessions
  infill.py          [[marked span]] repair CLI and JSON proof reporter
  ngram.py           local scratch n-gram and bridge guide for optional sampling scaffolds
  eval.py            masked-token evaluation and sampler smoke test
  bench.py           non-leaky validation benchmark with guide/retrieval/lattice/model baselines
  gate.py            benchmark-to-claim boundary checker
  export.py          slim checkpoint exporter
  verify_scratch.py  shortcut scanner
configs/
  tiny.json
  base.json
  mac_sota.json
  large.json
tests/
  unittest coverage for tokenizer, diffusion, model, sampler, benchmark, exporter, gate, verifier
```

## Honest Model Card

The included checkpoints prove that the model trains from scratch and generates through iterative diffusion. They are not foundation models. The impressive part is the complete scratch-built diffusion-LM stack, the novel sampler/corruption/benchmark mechanics, the replayable 30k Mac run, and the clean path from laptop proof to larger pretraining. To make a genuinely strong public model, train `configs/mac_sota.json`, `configs/base.json`, or `configs/large.json` on a larger licensed corpus until `helixdiff-bench` clears its held-out gates.

## Included Data

`data/seed_corpus.txt` is a small hand-written seed corpus for laptop smoke tests. `data/tinyshakespeare.txt` is the public Tiny Shakespeare corpus commonly mirrored from Andrej Karpathy's char-rnn example data, included so users can run a larger-from-scratch proof without hunting for a dataset.
