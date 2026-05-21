from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .data import load_text
from .ngram import BigramGuide
from .sample import choose_device, denoise_ids, load_checkpoint, render_tokens_with_masks
from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class MarkedInfill:
    tokens: torch.Tensor
    target: torch.Tensor
    frozen: torch.Tensor
    before: str
    hole: str
    after: str
    hole_start: int
    hole_end: int

    @property
    def hole_length(self) -> int:
        return self.hole_end - self.hole_start


def parse_marked_infill(marked_text: str, tokenizer: ByteTokenizer) -> MarkedInfill:
    """Parse one [[hole]] span and replace it with mask tokens of the same byte length."""
    if marked_text.count("[[") != 1 or marked_text.count("]]") != 1:
        raise ValueError("text must contain exactly one [[masked span]]")
    before, rest = marked_text.split("[[", 1)
    hole, after = rest.split("]]", 1)
    before_ids = tokenizer.encode(before, add_bos=True, add_eos=False)
    hole_ids = tokenizer.encode(hole, add_bos=False, add_eos=False)
    after_ids = tokenizer.encode(after, add_bos=False, add_eos=True)
    if not hole_ids:
        raise ValueError("masked span must contain at least one byte")
    target_ids = before_ids + hole_ids + after_ids
    masked_ids = before_ids + [tokenizer.mask_token_id] * len(hole_ids) + after_ids
    frozen = [True] * len(before_ids) + [False] * len(hole_ids) + [True] * len(after_ids)
    hole_start = len(before_ids)
    return MarkedInfill(
        tokens=torch.tensor(masked_ids, dtype=torch.long),
        target=torch.tensor(target_ids, dtype=torch.long),
        frozen=torch.tensor(frozen, dtype=torch.bool),
        before=before,
        hole=hole,
        after=after,
        hole_start=hole_start,
        hole_end=hole_start + len(hole_ids),
    )


def _hole_accuracy(predicted: torch.Tensor, target: torch.Tensor, start: int, end: int) -> float:
    pred_hole = predicted[start:end]
    target_hole = target[start:end]
    if target_hole.numel() == 0:
        return 0.0
    return float((pred_hole == target_hole).float().mean().item())


@torch.no_grad()
def run_infill(
    *,
    checkpoint: str | Path,
    marked_text: str,
    steps: int = 48,
    temperature: float = 0.85,
    top_k: int = 64,
    remask: float = 0.06,
    guide_data: str | None = None,
    guidance: float = 0.0,
    schedule: str = "entropy",
    seed: int = 7,
    device: str = "auto",
    max_reveal_per_step: int | None = 1,
) -> dict[str, Any]:
    torch_device = choose_device(device)
    model, tokenizer, payload = load_checkpoint(checkpoint, device=torch_device)
    example = parse_marked_infill(marked_text, tokenizer)
    if example.tokens.numel() > model.config.seq_len:
        raise ValueError(f"infill example has {example.tokens.numel()} tokens; checkpoint seq_len is {model.config.seq_len}")
    guide = None
    if guide_data and guidance > 0:
        guide = BigramGuide.from_text(load_text(guide_data), tokenizer).to_device(torch_device)
    started = time.perf_counter()
    repaired_ids, trace = denoise_ids(
        model,
        tokenizer,
        initial_tokens=example.tokens,
        frozen=example.frozen,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        remask=remask,
        allow_eos=False,
        guide=guide,
        guidance=guidance,
        schedule=schedule,
        seed=seed,
        return_trace=True,
        trace_preview=True,
        max_reveal_per_step=max_reveal_per_step,
    )
    elapsed = time.perf_counter() - started
    assert isinstance(repaired_ids, torch.Tensor)
    predicted_hole = tokenizer.decode(repaired_ids[example.hole_start : example.hole_end])
    target_text = tokenizer.decode(example.target)
    repaired_text = tokenizer.decode(repaired_ids)
    masked_text = render_tokens_with_masks(example.tokens, tokenizer)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "mechanism": "suture-trace infill",
        "marked_text": marked_text,
        "masked_text": masked_text,
        "target_text": target_text,
        "repaired_text": repaired_text,
        "target_hole": example.hole,
        "predicted_hole": predicted_hole,
        "hole_byte_accuracy": _hole_accuracy(repaired_ids, example.target, example.hole_start, example.hole_end),
        "hole_length_bytes": example.hole_length,
        "frozen_context_unchanged": bool((repaired_ids[example.frozen] == example.target[example.frozen]).all().item()),
        "steps_requested": steps,
        "steps_used": len(trace),
        "max_reveal_per_step": max_reveal_per_step,
        "elapsed_seconds": elapsed,
        "tokens_per_second": int(example.tokens.numel()) / max(elapsed, 1e-9),
        "guide_data": guide_data,
        "guidance": guidance,
        "schedule": schedule,
        "seed": seed,
        "trace": trace,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Repair one [[marked]] text span with the HelixDiff denoiser.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--text",
        default="The model begins with a [[sentence]], removes bytes until the page looks damaged.",
        help="Text containing exactly one [[span]] to mask and repair.",
    )
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--remask", type=float, default=0.06)
    parser.add_argument("--guide-data")
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--schedule", choices=["entropy", "ribbon"], default="entropy")
    parser.add_argument("--max-reveal-per-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    report = run_infill(
        checkpoint=args.checkpoint,
        marked_text=args.text,
        steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        remask=args.remask,
        guide_data=args.guide_data,
        guidance=args.guidance,
        schedule=args.schedule,
        seed=args.seed,
        device=args.device,
        max_reveal_per_step=args.max_reveal_per_step,
    )
    print(json.dumps(report, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
