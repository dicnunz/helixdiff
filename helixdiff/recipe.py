from __future__ import annotations

import argparse
import json
import shlex
from typing import Any

from .gate import STRICT_REPAIR_RECIPE


DEFAULT_CHECKPOINT = "checkpoints/helixdiff_tiny_shakespeare_clock_suture_30k_slim.pt"
DEFAULT_DATA = "data/tinyshakespeare.txt"
DEFAULT_BASELINE = "proof/bench_shakespeare_4k_unseen_candidates.json"
DEFAULT_ORACLE_OUT = "proof/lattice_oracle_strict_repair_8case.json"
DEFAULT_BENCH_OUT = "proof/bench_strict_repair_lattice_8case.json"
DEFAULT_CALIBRATION_OUT = "proof/selector_margin_calibration_strict_repair_8case.json"
DEFAULT_SELECTOR_CONTRACT_OUT = "proof/selector_contract_strict_repair_8case.json"
DEFAULT_GATE_OUT = "proof/gate_strict_repair_lattice_8case.json"


def comma_join(values: list[Any]) -> str:
    return ",".join(f"{float(value):g}" if isinstance(value, float) else str(value) for value in values)


def build_strict_repair_commands(
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    data: str = DEFAULT_DATA,
    baseline: str = DEFAULT_BASELINE,
    oracle_out: str = DEFAULT_ORACLE_OUT,
    bench_out: str = DEFAULT_BENCH_OUT,
    calibration_out: str = DEFAULT_CALIBRATION_OUT,
    selector_contract_out: str = DEFAULT_SELECTOR_CONTRACT_OUT,
    gate_out: str = DEFAULT_GATE_OUT,
    cases: int = 8,
    span_chars: int = 4,
    context_chars: int = 36,
    guidance: float = 0.2,
    steps: int = 48,
    top_k: int = 48,
    temperature: float = 0.55,
) -> dict[str, Any]:
    """Return the exact command sequence for the predeclared strict repair proof."""

    oracle = [
        "uv",
        "run",
        "helixdiff-bench",
        "--data",
        data,
        "--candidate-oracle-only",
        "--cases",
        str(cases),
        "--span-chars",
        str(span_chars),
        "--context-chars",
        str(context_chars),
        "--require-unseen-hole",
        "--lattice-bi-anchor-candidates",
        str(STRICT_REPAIR_RECIPE["lattice_bi_anchor_candidates"]),
        "--lattice-bi-anchor-sizes",
        comma_join(STRICT_REPAIR_RECIPE["lattice_bi_anchor_sizes"]),
        "--lattice-selector-anchor",
        str(STRICT_REPAIR_RECIPE["lattice_selector_anchor"]),
        "--lattice-local-surface-anchor-calibration",
        "--json-out",
        oracle_out,
    ]
    benchmark = [
        "uv",
        "run",
        "helixdiff-bench",
        "--checkpoint",
        checkpoint,
        "--data",
        data,
        "--cases",
        str(cases),
        "--span-chars",
        str(span_chars),
        "--context-chars",
        str(context_chars),
        "--require-unseen-hole",
        "--guidance",
        f"{guidance:g}",
        "--steps",
        str(steps),
        "--top-k",
        str(top_k),
        "--temperature",
        f"{temperature:g}",
        "--lattice-prior-rerank-top-k",
        str(STRICT_REPAIR_RECIPE["lattice_prior_rerank_top_k"]),
        "--lattice-verifier-mode",
        str(STRICT_REPAIR_RECIPE["lattice_verifier_mode"]),
        "--lattice-verifier-top-k",
        str(STRICT_REPAIR_RECIPE["lattice_verifier_top_k"]),
        "--lattice-selector-margin",
        f"{float(STRICT_REPAIR_RECIPE['lattice_selector_margin']):g}",
        "--lattice-selector-anchor",
        str(STRICT_REPAIR_RECIPE["lattice_selector_anchor"]),
        "--lattice-selector-anchor-sweep",
        comma_join(STRICT_REPAIR_RECIPE["lattice_selector_anchor_sweep"]),
        "--lattice-selector-margin-sweep",
        comma_join(STRICT_REPAIR_RECIPE["lattice_selector_margin_sweep"]),
        "--lattice-bi-anchor-candidates",
        str(STRICT_REPAIR_RECIPE["lattice_bi_anchor_candidates"]),
        "--lattice-bi-anchor-sizes",
        comma_join(STRICT_REPAIR_RECIPE["lattice_bi_anchor_sizes"]),
        "--lattice-local-surface-anchor-calibration",
        "--lattice-visible-reranker-calibration",
        "--json-out",
        bench_out,
    ]
    calibrate = [
        "uv",
        "run",
        "helixdiff-calibrate-selector",
        bench_out,
        "--json-out",
        calibration_out,
    ]
    selector_contract = [
        "uv",
        "run",
        "helixdiff-selector-contract",
        calibration_out,
        "--json-out",
        selector_contract_out,
    ]
    gate = [
        "uv",
        "run",
        "helixdiff-gate",
        "--current",
        bench_out,
        "--baseline",
        baseline,
        "--require-repair-proof-contract",
        "--json-out",
        gate_out,
    ]
    return {
        "kind": "helixdiff_strict_repair_proof_recipe",
        "claim_boundary": "narrow repair-lattice claim only if gate passes; never model SOTA",
        "strict_repair_recipe": STRICT_REPAIR_RECIPE,
        "artifacts": {
            "oracle": oracle_out,
            "benchmark": bench_out,
            "calibration": calibration_out,
            "selector_contract": selector_contract_out,
            "gate": gate_out,
        },
        "commands": {
            "oracle": oracle,
            "benchmark": benchmark,
            "calibrate": calibrate,
            "selector_contract": selector_contract,
            "gate": gate,
        },
        "shell_commands": {
            "oracle": shlex.join(oracle),
            "benchmark": shlex.join(benchmark),
            "calibrate": shlex.join(calibrate),
            "selector_contract": shlex.join(selector_contract),
            "gate": shlex.join(gate),
        },
        "heavy_slot_required": True,
        "oracle_heavy_slot_required": False,
        "notes": [
            "Run oracle first when the Mac is hot; it does not load the checkpoint.",
            "Claim the shared heavy slot before running benchmark.",
            "Run calibrate and gate after benchmark completes.",
            "Emit a selector contract after calibration; use it only as a frozen predeclaration for a later held-out run.",
            "Do not add lattice-local-prior-calibration to this first strict proof.",
            "Visible-reranker calibration is diagnostic-only here; do not apply it until a later held-out split.",
            "Bi-anchor gap candidates are train-split-only; oracle containment is not model accuracy.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the strict HelixDiff repair-lattice proof recipe.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--oracle-out", default=DEFAULT_ORACLE_OUT)
    parser.add_argument("--bench-out", default=DEFAULT_BENCH_OUT)
    parser.add_argument("--calibration-out", default=DEFAULT_CALIBRATION_OUT)
    parser.add_argument("--selector-contract-out", default=DEFAULT_SELECTOR_CONTRACT_OUT)
    parser.add_argument("--gate-out", default=DEFAULT_GATE_OUT)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--span-chars", type=int, default=4)
    parser.add_argument("--context-chars", type=int, default=36)
    parser.add_argument("--guidance", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)
    recipe = build_strict_repair_commands(
        checkpoint=args.checkpoint,
        data=args.data,
        baseline=args.baseline,
        oracle_out=args.oracle_out,
        bench_out=args.bench_out,
        calibration_out=args.calibration_out,
        selector_contract_out=args.selector_contract_out,
        gate_out=args.gate_out,
        cases=args.cases,
        span_chars=args.span_chars,
        context_chars=args.context_chars,
        guidance=args.guidance,
        steps=args.steps,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    if args.json:
        print(json.dumps(recipe, indent=2))
        return
    print("# Strict HelixDiff repair-lattice proof")
    for name in ("oracle", "benchmark", "calibrate", "selector_contract", "gate"):
        print()
        print(f"## {name}")
        print(recipe["shell_commands"][name])


if __name__ == "__main__":
    main()
