from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


BANNED_SNIPPETS = [
    "from" + "_pretrained",
    "Auto" + "Model",
    "Auto" + "Tokenizer",
    "pipeline" + "(",
    "huggingface" + "_hub",
    "transform" + "ers",
    "open" + "ai.",
    "anthropic" + ".",
    "ollama" + ".chat",
]

SCAN_SUFFIXES = {".py", ".toml"}


def scan_repo(root: Path) -> list[dict[str, str | int]]:
    findings: list[dict[str, str | int]] = []
    for path in root.rglob("*"):
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        if path.suffix not in SCAN_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for snippet in BANNED_SNIPPETS:
                if snippet in line:
                    findings.append({"path": str(path.relative_to(root)), "line": line_no, "snippet": snippet})
    return findings


def check_checkpoint(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"checked": False}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ok = bool(payload.get("scratch_only")) and not bool(payload.get("pretrained_weights"))
    return {
        "checked": True,
        "path": str(path),
        "scratch_only": bool(payload.get("scratch_only")),
        "pretrained_weights": bool(payload.get("pretrained_weights")),
        "hosted_model_calls": bool(payload.get("hosted_model_calls")),
        "checkpoint_step": payload.get("step"),
        "ok": ok,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify HelixDiff did not use pretrained-model shortcuts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--checkpoint")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    findings = scan_repo(root)
    checkpoint_report = check_checkpoint(Path(args.checkpoint).resolve() if args.checkpoint else None)
    report = {
        "root": str(root),
        "banned_snippet_findings": findings,
        "checkpoint": checkpoint_report,
        "ok": not findings and (not args.checkpoint or bool(checkpoint_report.get("ok"))),
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
