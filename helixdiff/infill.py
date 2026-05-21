from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .data import load_text
from .ngram import BigramGuide
from .sample import _filter_logits, choose_device, denoise_ids, load_checkpoint, render_tokens_with_masks
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


def _sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
def score_repair(
    *,
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    repaired_ids: torch.Tensor,
    hole_start: int,
    hole_end: int,
    guide: BigramGuide | None,
    guidance: float,
    temperature: float,
    top_k: int,
    mode: str = "suture_loo",
) -> float:
    device = next(model.parameters()).device
    candidate = repaired_ids.to(device=device, dtype=torch.long).unsqueeze(0)
    hole_len = max(0, hole_end - hole_start)
    if mode == "full_hole" or hole_len <= 1:
        probes = candidate.clone()
        probes[:, hole_start:hole_end] = tokenizer.mask_token_id
        target_positions = torch.arange(hole_start, hole_end, device=device)
        targets = candidate[:, hole_start:hole_end].expand(probes.shape[0], -1)
        t = torch.full((probes.shape[0],), 0.25, device=device)
    elif mode == "suture_loo":
        probes = candidate.repeat(hole_len, 1)
        target_positions = torch.arange(hole_start, hole_end, device=device)
        probes[torch.arange(hole_len, device=device), target_positions] = tokenizer.mask_token_id
        targets = candidate[:, hole_start:hole_end]
        t = torch.full((probes.shape[0],), 0.08, device=device)
    else:
        raise ValueError(f"unknown repair score mode: {mode}")
    mask_fraction = (probes == tokenizer.mask_token_id).float().mean(dim=1)
    logits = model(probes, t, corruption_mode=3, mask_fraction=mask_fraction)
    if guide is not None and guidance > 0:
        logits = logits + guidance * guide.logits(probes, tokenizer)
    logits = _filter_logits(
        logits / max(temperature, 1e-4),
        tokenizer,
        top_k=top_k,
        allowed_token_ids=getattr(model, "allowed_token_ids", None),
        allow_eos=False,
    )
    if mode == "suture_loo" and hole_len > 1:
        row_ids = torch.arange(hole_len, device=device)
        log_probs = torch.log_softmax(logits[row_ids, target_positions, :], dim=-1)
        token_scores = log_probs.gather(-1, targets.reshape(-1, 1)).squeeze(-1)
    else:
        log_probs = torch.log_softmax(logits[:, hole_start:hole_end, :], dim=-1)
        token_scores = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).reshape(-1)
    return float(token_scores.mean().item())


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
    candidates: int = 1,
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
    best_ids: torch.Tensor | None = None
    best_trace: list[dict[str, Any]] = []
    best_score = float("-inf")
    candidate_rows: list[dict[str, Any]] = []
    for offset in range(max(1, candidates)):
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
            seed=seed + offset,
            return_trace=True,
            trace_preview=True,
            max_reveal_per_step=max_reveal_per_step,
            corruption_mode=3,
        )
        assert isinstance(repaired_ids, torch.Tensor)
        score = score_repair(
            model=model,
            tokenizer=tokenizer,
            repaired_ids=repaired_ids,
            hole_start=example.hole_start,
            hole_end=example.hole_end,
            guide=guide,
            guidance=guidance,
            temperature=temperature,
            top_k=top_k,
        )
        candidate_rows.append(
            {
                "seed": seed + offset,
                "score": score,
                "predicted_hole": tokenizer.decode(repaired_ids[example.hole_start : example.hole_end]),
                "hole_byte_accuracy": _hole_accuracy(repaired_ids, example.target, example.hole_start, example.hole_end),
            }
        )
        if score > best_score:
            best_score = score
            best_ids = repaired_ids
            best_trace = trace
    elapsed = time.perf_counter() - started
    if best_ids is None:
        raise RuntimeError("no infill candidates were produced")
    repaired_ids = best_ids
    predicted_hole = tokenizer.decode(repaired_ids[example.hole_start : example.hole_end])
    target_text = tokenizer.decode(example.target)
    repaired_text = tokenizer.decode(repaired_ids)
    masked_text = render_tokens_with_masks(example.tokens, tokenizer)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
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
        "steps_used": len(best_trace),
        "max_reveal_per_step": max_reveal_per_step,
        "candidates": max(1, candidates),
        "selected_candidate_score": best_score,
        "candidate_summaries": candidate_rows,
        "elapsed_seconds": elapsed,
        "tokens_per_second": int(example.tokens.numel()) / max(elapsed, 1e-9),
        "guide_data": guide_data,
        "guidance": guidance,
        "schedule": schedule,
        "seed": seed,
        "trace": best_trace,
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
    parser.add_argument("--candidates", type=int, default=1)
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
        candidates=args.candidates,
    )
    print(json.dumps(report, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
