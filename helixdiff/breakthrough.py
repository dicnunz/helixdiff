from __future__ import annotations

import argparse
import json
from typing import Any

from .recipe import build_strict_repair_commands


SOURCE_REFRESH_DATE = "2026-05-21"
CLAIM_BOUNDARY = (
    "Plausible SOTA target is narrow Mac-local visible-context document repair, "
    "not global language-model SOTA."
)

DIFFUSION_LM_SOURCES: list[dict[str, str]] = [
    {
        "id": "arxiv:2406.07524",
        "title": "Simple and Effective Masked Diffusion Language Models",
        "url": "https://arxiv.org/abs/2406.07524",
        "usable_signal": "clean masked-diffusion objectives and samplers can reach SOTA among diffusion LMs",
    },
    {
        "id": "arxiv:2503.09573",
        "title": "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models",
        "url": "https://arxiv.org/abs/2503.09573",
        "usable_signal": "block locality, arbitrary length, efficient training, and data-driven noise schedules matter",
    },
    {
        "id": "arxiv:2310.16834",
        "title": "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution",
        "url": "https://arxiv.org/abs/2310.16834",
        "usable_signal": "discrete score/ratio objectives improve diffusion LM likelihood and controllable infilling",
    },
    {
        "id": "arxiv:2502.09992",
        "title": "Large Language Diffusion Models",
        "url": "https://arxiv.org/abs/2502.09992",
        "usable_signal": "scratch diffusion LMs can scale, but that evidence is an 8B-scale boundary warning for HelixDiff",
    },
    {
        "id": "arxiv:2510.18114",
        "title": "Latent-Augmented Discrete Diffusion Models",
        "url": "https://arxiv.org/abs/2510.18114",
        "usable_signal": "factored reverse transitions lose cross-token structure; add cheap joint signals before few-step repair",
    },
    {
        "id": "aclanthology:2025.babylm-main.38",
        "title": "Masked Diffusion Language Models with Frequency-Informed Training",
        "url": "https://aclanthology.org/2025.babylm-main.38.pdf",
        "usable_signal": "data-constrained diffusion training can prioritize rare or hard tokens without leaving the diffusion frame",
    },
    {
        "id": "arxiv:2604.03677",
        "title": "Unlocking Prompt Infilling Capability for Diffusion Language Models",
        "url": "https://arxiv.org/abs/2604.03677",
        "usable_signal": "full-sequence masking practice can unlock infilling; HelixDiff's suture curriculum should treat masking policy as a first-class lever",
    },
    {
        "id": "arxiv:2602.01326",
        "title": "DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas",
        "url": "https://arxiv.org/abs/2602.01326",
        "usable_signal": "fixed-length canvases are a known practical infilling blocker; HelixDiff must disclose fixed-span limits until variable-length repair is implemented",
    },
    {
        "id": "arxiv:2602.15014",
        "title": "Scaling Beyond Masked Diffusion Language Models",
        "url": "https://arxiv.org/abs/2602.15014",
        "usable_signal": "perplexity is not enough across diffusion families; narrow claims should report repair accuracy, baselines, and speed-quality tradeoffs",
    },
    {
        "id": "arxiv:2506.23529",
        "title": "When Test-Time Adaptation Meets Self-Supervised Models",
        "url": "https://arxiv.org/abs/2506.23529",
        "usable_signal": "test-time adaptation on self-supervised signals needs an explicit protocol and collaboration/consistency checks, not blind local updates",
    },
]


def build_breakthrough_plan() -> dict[str, Any]:
    strict_recipe = build_strict_repair_commands()
    oracle_command = strict_recipe["shell_commands"]["oracle"]
    visible_smoke_command = (
        "uv run helixdiff-visible-reranker-oracle-smoke "
        "--config configs/proof_visible_reranker_oracle_smoke.yaml "
        "--out proof/visible_reranker_oracle_smoke.json --json"
    )
    proxy_mask_contract_command = (
        "uv run helixdiff-proxy-mask-selector-contract-smoke "
        "--config configs/proof_proxy_mask_selector_contract_smoke.yaml "
        "--out proof/proxy_mask_selector_contract_smoke.json "
        "--contract-out proof/proxy_mask_selector_contract_smoke_contract.json --json"
    )
    benchmark_command = strict_recipe["shell_commands"]["benchmark"]
    calibrate_command = strict_recipe["shell_commands"]["calibrate"]
    selector_contract_command = strict_recipe["shell_commands"]["selector_contract"]
    gate_command = strict_recipe["shell_commands"]["gate"]
    heldout_contract_command = benchmark_command.replace(
        "--json-out proof/bench_strict_repair_lattice_8case.json",
        (
            "--seed 202 "
            "--lattice-selector-contract proof/selector_contract_strict_repair_8case.json "
            "--lattice-require-selector-contract-ready "
            "--json-out proof/bench_selector_contract_heldout_8case.json"
        ),
    )
    lanes = [
        {
            "rank": 1,
            "name": "visible_reranker_oracle_smoke",
            "thesis": "Before claiming a reranker breakthrough, prove it is model-free, diagnostic-only, leakage-audited, and causally tied to visible context.",
            "source_ids": ["arxiv:2310.16834", "arxiv:2510.18114"],
            "repo_move": "Emit the visible-reranker oracle smoke receipt with shuffle and counterfactual-context falsification before any heavy reranker claim.",
            "proof_commands": [visible_smoke_command],
            "pass_condition": "receipt has prior/surface/visible_reranker branches, calibration, apply=false, leakage audit, shuffle falsification, and causal-visible counterfactual context audit",
            "claim_if_passes": "model-free visible-context reranker proof only",
            "kill_condition": "reranker is applied, labels choose the branch, shuffle/counterfactual context does not collapse, or leakage flags fire",
            "heavy_slot_required": False,
        },
        {
            "rank": 2,
            "name": "proxy_mask_selector_contract_smoke",
            "thesis": "Choose a selector contract from visible proxy masks before target scoring, and make proxy masks target-shaped without target leakage.",
            "source_ids": ["arxiv:2310.16834", "arxiv:2510.18114"],
            "repo_move": "Emit a ready boundary-only selector contract plus fail-closed target-shadow and target-geometry proxy probes.",
            "proof_commands": [proxy_mask_contract_command],
            "pass_condition": (
                "receipt has contract_ready=true, target_metric_used_for_selection=false, pseudo-heldout beats shuffle, "
                "target metrics remain diagnostic, and claim_gate blocks target-lift wording unless useful_ratchet=true"
            ),
            "claim_if_passes": "visible-only selector predeclaration smoke only",
            "kill_condition": (
                "proxy heldout fails shuffle, target gold leaks into selection, target metrics are used to choose the contract, "
                "target-retrieval geometry or target-shadow proxies fail their pseudo-heldout shuffle checks, "
                "or --require-useful-ratchet fails for any target-lift claim"
            ),
            "heavy_slot_required": False,
        },
        {
            "rank": 3,
            "name": "gold_blind_bi_anchor_oracle",
            "thesis": "If the exact span is not in the train-only visible-anchor lattice, no sampler can rescue the run.",
            "source_ids": ["arxiv:2310.16834", "arxiv:2510.18114"],
            "repo_move": "Run the no-checkpoint oracle from helixdiff-proof-recipe before any heavy model proof.",
            "proof_commands": [oracle_command],
            "pass_condition": (
                "candidate_oracle reports fixed-K exact-span coverage, bi-anchor contribution, split hashes, "
                "and zero train/eval same-hole leakage before model scoring"
            ),
            "claim_if_passes": "candidate-lattice coverage only; not model accuracy",
            "kill_condition": "oracle coverage does not lift, K explodes, or coverage depends on leaked validation text",
            "heavy_slot_required": False,
        },
        {
            "rank": 4,
            "name": "strict_repair_lattice_proof",
            "thesis": "Make the next benchmark a predeclared trial, not a sampler search.",
            "source_ids": ["arxiv:2406.07524", "arxiv:2503.09573"],
            "repo_move": "Run the exact helixdiff-proof-recipe benchmark, calibrate, freeze the selector, then gate.",
            "proof_commands": [benchmark_command, calibrate_command, selector_contract_command, gate_command],
            "pass_condition": (
                "retrieval_lattice beats bridge-only and nearest-visible byte/exact baselines "
                "while --require-repair-proof-contract passes"
            ),
            "claim_if_passes": "narrow repair-lattice claim only",
            "kill_condition": "case_count, proof contract, nearest-visible lift, or frozen-context preservation fails",
            "heavy_slot_required": True,
        },
        {
            "rank": 5,
            "name": "frozen_selector_heldout_trial",
            "thesis": "A selector choice is not evidence until a separate benchmark loads it from a ready contract.",
            "source_ids": ["arxiv:2310.16834", "arxiv:2510.18114"],
            "repo_move": "Apply --lattice-selector-contract only after helixdiff-selector-contract is ready_for_heldout.",
            "proof_commands": [heldout_contract_command],
            "pass_condition": "held-out report records selector-contract id/hash and the repair gate accepts it",
            "claim_if_passes": "predeclared selector contract improved held-out repair selection",
            "kill_condition": "contract is diagnostic-only, missing hash/id, or held-out lift disappears",
            "heavy_slot_required": True,
        },
        {
            "rank": 6,
            "name": "visible_hole_reranker",
            "thesis": "The answer is often in a tiny top-k set; the breakthrough is selecting it without hidden leakage.",
            "source_ids": ["arxiv:2310.16834", "arxiv:2510.18114"],
            "repo_move": (
                "Use the calibrator's visible_hole_reranker receipt to confirm oracle-in-scored-set "
                "with raw verifier misses, then test the visible-context top-k reranker anchor "
                "before spending compute on heavier model adaptation."
            ),
            "proof_commands": [
                "uv run helixdiff-calibrate-selector proof/bench_prior_topk_dual_smoke.json --min-cases 1",
                "uv run helixdiff-bench --help | rg 'visible-reranker|visible_reranker'",
                oracle_command,
                "uv run helixdiff-calibrate-selector proof/bench_strict_repair_lattice_8case.json --json-out proof/selector_margin_calibration_strict_repair_8case.json",
                "uv run helixdiff-selector-contract proof/selector_margin_calibration_strict_repair_8case.json --json-out proof/selector_contract_strict_repair_8case.json --require-ready",
            ],
            "pass_condition": "raw verifier misses are rescued without lowering oracle-in-scored-set coverage",
            "claim_if_passes": "verifier-guided lattice selection improved a held-out repair benchmark",
            "kill_condition": "visible-hole verifier overfits synthetic holes or harms prior exact hits",
            "heavy_slot_required": False,
        },
        {
            "rank": 7,
            "name": "prompt_canvas_curriculum",
            "thesis": "The next training breakthrough is not just more masking; it is matching the masking canvas to visible-context repair.",
            "source_ids": ["arxiv:2604.03677", "arxiv:2602.01326", "arxiv:2602.15014"],
            "repo_move": (
                "Keep the public benchmark fixed-span for now, then add a separate variable-length canvas gate before claiming "
                "code infilling or general flexible infill. Inside fixed-span repair, continue optimizing full-context suture masking."
            ),
            "proof_commands": [
                "uv run helixdiff-breakthrough-plan --json",
                "uv run helixdiff-proof-recipe --json",
            ],
            "pass_condition": "release plan discloses fixed-length canvas limits and the proof recipe reports repair accuracy against baselines, not perplexity alone",
            "claim_if_passes": "fixed-span visible-context repair plan is source-backed and canvas-boundary-aware",
            "kill_condition": "any README/release language claims flexible-length infilling, code-infilling SOTA, or broad diffusion-LM quality without a variable-length gate",
            "heavy_slot_required": False,
        },
        {
            "rank": 8,
            "name": "frequency_block_suture_curriculum",
            "thesis": "Mac-local training must spend scarce gradient signal on rare boundary bytes and local blocks.",
            "source_ids": ["arxiv:2503.09573", "aclanthology:2025.babylm-main.38", "arxiv:2604.03677"],
            "repo_move": (
                "Add a next config that combines block-local corruption, boundary-pinned suture spans, "
                "and frequency-aware sampling for rare span-edge bytes."
            ),
            "proof_commands": [
                "uv run helixdiff-train --config configs/mac_sota.json --data data/tinyshakespeare.txt --steps 100000 --checkpoint checkpoints/helixdiff_mac_sota.pt",
                "uv run helixdiff-bench --checkpoint checkpoints/helixdiff_mac_sota.pt --data data/tinyshakespeare.txt --require-unseen-hole --json-out proof/bench_mac_sota.json",
            ],
            "pass_condition": "model-only infill beats bridge-only before any retrieval-lattice claim",
            "claim_if_passes": "small Mac-trained checkpoint improved model-quality gate",
            "kill_condition": "masked accuracy or model-only infill fails the gate",
            "heavy_slot_required": True,
        },
        {
            "rank": 9,
            "name": "latent_surface_signature",
            "thesis": "A tiny model needs cheap joint structure; surface-unit signatures can mimic some latent-channel benefits.",
            "source_ids": ["arxiv:2510.18114", "arxiv:2502.09992"],
            "repo_move": (
                "Attach non-leaky byte-class, word-shape, speaker-label, and dash-bridge signatures to lattice rows "
                "before verifier scoring."
            ),
            "proof_commands": [
                "uv run python -m unittest tests.test_bench tests.test_gate tests.test_calibrate",
            ],
            "pass_condition": "failure taxonomy moves from answer_absent to selector_rescued on held-out cases",
            "claim_if_passes": "surface-latent candidate features improved the repair selector",
            "kill_condition": "features leak validation spans or only duplicate nearest-visible retrieval",
            "heavy_slot_required": False,
        },
    ]
    return {
        "kind": "helixdiff_mac_local_breakthrough_plan",
        "source_refresh_date": SOURCE_REFRESH_DATE,
        "chatgpt_teammate_status": {
            "requested": True,
            "usable_this_run": True,
            "model_mode": "Extended Pro",
            "conversation_url": "https://chatgpt.com/c/6a0f1cf7-b734-83ea-afa1-1a92152682f1",
            "latest_request": "fresh Extended Pro critique for target-shadow proxy calibration",
            "latest_response_status": "answered",
            "latest_recommendation": "target-shadow proxy calibration",
            "blocker": None,
            "claim": (
                "Extended Pro recommended target-shadow proxy calibration: match visible pseudo masks "
                "to the redacted target lattice fingerprint before freezing a selector."
            ),
            "recorded_prior_note": {
                "model_mode": "Extended Pro",
                "conversation_url": "https://chatgpt.com/c/6a0f1cf7-b734-83ea-afa1-1a92152682f1",
                "contribution": (
                    "Recorded prior note: test target-shadow proxy calibration after the proxy-mask "
                    "selector failed to transfer from generic visible pseudo holes."
                ),
                "verification_status": "current_chrome_readback_reverified_url",
            },
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "current_best_move": "proxy_mask_selector_contract_smoke",
        "sources": DIFFUSION_LM_SOURCES,
        "lanes": lanes,
        "release_standard": [
            "no pretrained weights",
            "no paid API calls in shipped runner",
            "held-out bytes excluded from adaptation and calibration examples",
            "frozen visible context preserved exactly",
            "bridge-only and nearest-visible baselines reported",
            "proxy-mask receipts cannot be cited as target-lift evidence unless --require-useful-ratchet passes",
            "fixed-length canvas limits disclosed until a variable-length repair gate exists",
            "global model-quality language forbidden unless helixdiff-gate model_quality_passed is true",
        ],
    }


def _format_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# HelixDiff Mac-Local Breakthrough Plan",
        "",
        f"Source refresh: `{plan['source_refresh_date']}`",
        "Evaluator: future maintainer deciding what to run next on a hot Mac.",
        "",
        f"Claim boundary: {plan['claim_boundary']}",
        "",
        "## Browser Teammate",
        "",
        f"- GPT-5.5 requested: `{plan['chatgpt_teammate_status']['requested']}`",
        f"- Usable this run: `{plan['chatgpt_teammate_status']['usable_this_run']}`",
    ]
    if plan["chatgpt_teammate_status"]["usable_this_run"]:
        lines.extend(
            [
                f"- Model/mode: {plan['chatgpt_teammate_status'].get('model_mode', 'unverified')}",
                f"- Conversation: {plan['chatgpt_teammate_status'].get('conversation_url', 'none')}",
                f"- Contribution: {plan['chatgpt_teammate_status']['claim']}",
            ]
        )
    else:
        lines.extend(
            [
                f"- Blocker: {plan['chatgpt_teammate_status'].get('blocker', 'unverified')}",
                f"- Claim: {plan['chatgpt_teammate_status']['claim']}",
            ]
        )
        prior_note = plan["chatgpt_teammate_status"].get("recorded_prior_note")
        if isinstance(prior_note, dict):
            lines.extend(
                [
                    f"- Recorded prior mode: {prior_note.get('model_mode', 'unknown')}",
                    f"- Recorded prior URL: {prior_note.get('conversation_url', 'none')}",
                    f"- Recorded prior contribution: {prior_note.get('contribution', 'none')}",
                    f"- Recorded prior status: {prior_note.get('verification_status', 'unknown')}",
                ]
            )
    lines.extend(
        [
            "",
            "## Decision Table",
            "",
            "| Rank | Lane | Why It Matters | First Move | Gate | Cost |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for lane in plan["lanes"]:
        cost = "heavy" if lane["heavy_slot_required"] else "cheap"
        lines.append(
            "| {rank} | `{name}` | {thesis} | {repo_move} | {pass_condition} | {cost} |".format(
                rank=lane["rank"],
                name=lane["name"],
                thesis=lane["thesis"],
                repo_move=lane["repo_move"],
                pass_condition=lane["pass_condition"],
                cost=cost,
            )
        )
    lines.extend(["", "## Command Ladder", ""])
    current_lane = next(
        lane for lane in plan["lanes"] if lane["name"] == plan["current_best_move"]
    )
    for index, command in enumerate(current_lane["proof_commands"], start=1):
        lines.append(f"{index}. `{command}`")
    lines.extend(["", "Cheap mutation if the heavy lane fails:"])
    cheap_lanes = [lane for lane in plan["lanes"] if not lane["heavy_slot_required"]]
    for lane in cheap_lanes:
        lines.append(f"- `{lane['name']}`: {lane['repo_move']} Stop if {lane['kill_condition']}.")
    lines.extend(["", "## Sources", ""])
    for source in plan["sources"]:
        lines.append(f"- `{source['id']}` [{source['title']}]({source['url']}): {source['usable_signal']}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the HelixDiff Mac-local breakthrough plan.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)
    plan = build_breakthrough_plan()
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(_format_markdown(plan))


if __name__ == "__main__":
    main()
