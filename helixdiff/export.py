from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .sample import load_checkpoint


REQUIRED_MECHANISMS = (
    "boundary_pinned_suture_corruption",
    "clock_conditioned_denoiser",
    "corruption_mode_conditioned_denoiser",
)


def _sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _merge_mechanisms(payload: dict[str, Any]) -> list[str]:
    mechanisms = list(payload.get("novel_mechanisms") or [])
    for mechanism in REQUIRED_MECHANISMS:
        if mechanism not in mechanisms:
            mechanisms.append(mechanism)
    return mechanisms


def export_checkpoint(
    checkpoint: str | Path,
    out: str | Path,
    *,
    use_ema: bool = True,
    verify_load: bool = True,
) -> dict[str, Any]:
    source = Path(checkpoint)
    target = Path(out)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state_key = "ema_model_state" if use_ema and payload.get("ema_model_state") else "model_state"
    exported = dict(payload)
    exported["model_state"] = payload[state_key]
    exported["ema_model_state"] = None
    exported["optimizer_state"] = None
    exported["exported_from"] = str(source)
    exported["exported_state"] = state_key
    exported["novel_mechanisms"] = _merge_mechanisms(payload)
    metrics = dict(exported.get("metrics") or {})
    metrics["exported_state"] = state_key
    exported["metrics"] = metrics
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, target)
    if verify_load:
        load_checkpoint(target, device="cpu", use_ema=False)
    return {
        "source": str(source),
        "source_sha256": _sha256_file(source),
        "out": str(target),
        "out_sha256": _sha256_file(target),
        "checkpoint_step": int(payload.get("step", 0)),
        "exported_state": state_key,
        "bytes": target.stat().st_size,
        "mechanisms": exported["novel_mechanisms"],
        "verify_load": bool(verify_load),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a slim HelixDiff checkpoint for GitHub/download use.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-weights", action="store_true", help="Export raw weights instead of EMA when EMA exists.")
    parser.add_argument("--no-verify-load", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    report = export_checkpoint(
        args.checkpoint,
        args.out,
        use_ema=not args.raw_weights,
        verify_load=not args.no_verify_load,
    )
    print(json.dumps(report, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
