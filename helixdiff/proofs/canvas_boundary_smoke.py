from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from helixdiff.breakthrough import build_breakthrough_plan
from helixdiff.recipe import build_strict_repair_commands


REQUIRED_SOURCE_IDS = {
    "arxiv:2604.03677",
    "arxiv:2602.01326",
    "arxiv:2602.15014",
}
FIXED_CANVAS_STANDARD = "fixed-length canvas limits disclosed until a variable-length repair gate exists"
VARIABLE_LENGTH_FLAGS = {
    "--variable-length",
    "--variable-span",
    "--length-states",
    "--canvas-expand",
    "--canvas-contract",
}


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _has_variable_length_gate(recipe: dict[str, Any]) -> bool:
    command_tokens: list[str] = []
    for command in recipe.get("commands", {}).values():
        if isinstance(command, list):
            command_tokens.extend(str(token) for token in command)
    return any(flag in command_tokens for flag in VARIABLE_LENGTH_FLAGS)


def build_receipt(
    *,
    readme_path: str | Path = "README.md",
    research_path: str | Path = "BREAKTHROUGH_RESEARCH_3H.md",
) -> dict[str, Any]:
    plan = build_breakthrough_plan()
    recipe = build_strict_repair_commands()
    readme = _read_text(readme_path)
    research = _read_text(research_path)
    source_ids = {source["id"] for source in plan["sources"]}
    lanes = {lane["name"]: lane for lane in plan["lanes"]}
    prompt_canvas_lane = lanes.get("prompt_canvas_curriculum", {})
    variable_length_gate_exists = _has_variable_length_gate(recipe)

    checks = {
        "prompt_infilling_sources_present": REQUIRED_SOURCE_IDS.issubset(source_ids),
        "prompt_canvas_lane_present": bool(prompt_canvas_lane),
        "prompt_canvas_lane_is_cheap": prompt_canvas_lane.get("heavy_slot_required") is False,
        "release_standard_discloses_fixed_canvas": FIXED_CANVAS_STANDARD in plan["release_standard"],
        "proof_recipe_is_fixed_span": "--span-chars" in recipe["commands"]["benchmark"],
        "proof_recipe_reports_repair_not_perplexity_only": "repair-lattice" in recipe["claim_boundary"],
        "readme_discloses_fixed_canvas_boundary": (
            "Current canvas boundary" in readme and "fixed-span visible-context repair" in readme
        ),
        "research_discloses_variable_length_gate": (
            "flexible-length code infilling needs a separate gate" in research
        ),
        "variable_length_gate_absent": not variable_length_gate_exists,
    }
    claim_allowed = {
        "fixed_span_visible_context_repair_plan": all(checks.values()),
        "flexible_length_or_code_infilling": False,
    }
    return {
        "proof_name": "canvas_boundary_smoke",
        "kind": "helixdiff_canvas_boundary_receipt",
        "claim_boundary": (
            "fixed-span visible-context repair only; flexible-length or code-infilling claims require "
            "a future variable-length repair gate"
        ),
        "sources": sorted(REQUIRED_SOURCE_IDS),
        "variable_length_flags_checked": sorted(VARIABLE_LENGTH_FLAGS),
        "variable_length_gate_exists": variable_length_gate_exists,
        "checks": checks,
        "claim_allowed": claim_allowed,
        "verdict": "pass" if all(checks.values()) else "fail",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Emit the HelixDiff fixed-canvas claim-boundary smoke receipt.")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--research", default="BREAKTHROUGH_RESEARCH_3H.md")
    parser.add_argument("--out")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--require-variable-length-gate",
        action="store_true",
        help="Fail unless a future variable-length repair gate is present.",
    )
    args = parser.parse_args(argv)
    receipt = build_receipt(readme_path=args.readme, research_path=args.research)
    if args.require_variable_length_gate and not receipt["variable_length_gate_exists"]:
        receipt["verdict"] = "fail"
        receipt["failure_reasons"] = ["variable_length_gate_required"]
    text = json.dumps(receipt, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text if args.json or not args.out else f"wrote {args.out}")
    if receipt["verdict"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
