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
]


def build_breakthrough_plan() -> dict[str, Any]:
    strict_recipe = build_strict_repair_commands()
    benchmark_command = strict_recipe["shell_commands"]["benchmark"]
    calibrate_command = strict_recipe["shell_commands"]["calibrate"]
    gate_command = strict_recipe["shell_commands"]["gate"]
    lanes = [
        {
            "rank": 1,
            "name": "strict_repair_lattice_proof",
            "thesis": "Make the next benchmark a predeclared trial, not a sampler search.",
            "source_ids": ["arxiv:2406.07524", "arxiv:2503.09573"],
            "repo_move": "Run the exact helixdiff-proof-recipe benchmark, then calibrate and gate.",
            "proof_commands": [benchmark_command, calibrate_command, gate_command],
            "pass_condition": (
                "retrieval_lattice beats bridge-only and nearest-visible byte/exact baselines "
                "while --require-repair-proof-contract passes"
            ),
            "claim_if_passes": "narrow repair-lattice claim only",
            "kill_condition": "case_count, proof contract, nearest-visible lift, or frozen-context preservation fails",
            "heavy_slot_required": True,
        },
        {
            "rank": 2,
            "name": "visible_hole_reranker",
            "thesis": "The answer is often in a tiny top-k set; the breakthrough is selecting it without hidden leakage.",
            "source_ids": ["arxiv:2310.16834", "arxiv:2510.18114"],
            "repo_move": (
                "Train or calibrate a tiny verifier only on synthetic holes from visible context, "
                "then rerank the structural-prior top-4 candidates."
            ),
            "proof_commands": [
                "uv run helixdiff-bench --candidate-oracle-only --cases 8 --require-unseen-hole --json-out proof/lattice_oracle_next_8case.json",
                "uv run helixdiff-calibrate-selector proof/bench_strict_repair_lattice_8case.json --json-out proof/selector_margin_calibration_strict_repair_8case.json",
            ],
            "pass_condition": "raw verifier misses are rescued without lowering oracle-in-scored-set coverage",
            "claim_if_passes": "verifier-guided lattice selection improved a held-out repair benchmark",
            "kill_condition": "visible-hole verifier overfits synthetic holes or harms prior exact hits",
            "heavy_slot_required": False,
        },
        {
            "rank": 3,
            "name": "frequency_block_suture_curriculum",
            "thesis": "Mac-local training must spend scarce gradient signal on rare boundary bytes and local blocks.",
            "source_ids": ["arxiv:2503.09573", "aclanthology:2025.babylm-main.38"],
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
            "rank": 4,
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
            "usable_this_run": False,
            "blocker": "Chrome/ChatGPT browser bridge returns Transport closed",
            "claim": "No GPT-5.5 Pro contribution is claimed unless a live ChatGPT transcript exists.",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "current_best_move": "strict_repair_lattice_proof",
        "sources": DIFFUSION_LM_SOURCES,
        "lanes": lanes,
        "release_standard": [
            "no pretrained weights",
            "no paid API calls in shipped runner",
            "held-out bytes excluded from adaptation and calibration examples",
            "frozen visible context preserved exactly",
            "bridge-only and nearest-visible baselines reported",
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
        f"- Blocker: {plan['chatgpt_teammate_status']['blocker']}",
        "",
        "## Decision Table",
        "",
        "| Rank | Lane | Why It Matters | First Move | Gate | Cost |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
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
    top_lane = plan["lanes"][0]
    for index, command in enumerate(top_lane["proof_commands"], start=1):
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
