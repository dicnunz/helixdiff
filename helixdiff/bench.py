from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from .data import ByteStream, load_text
from .diffusion import corrupt_batch, masked_accuracy, masked_cross_entropy, restrict_logits_to_ids
from .infill import parse_marked_infill, score_repair
from .ngram import BigramGuide
from .sample import choose_device, denoise_ids, load_checkpoint
from .tokenizer import ByteTokenizer


def split_text(text: str, val_fraction: float = 0.08) -> tuple[str, str]:
    split_at = max(1, int(len(text) * (1.0 - val_fraction)))
    return text[:split_at], text[split_at:]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def make_marked_cases(
    text: str,
    *,
    cases: int,
    span_chars: int,
    context_chars: int,
    seed: int,
    forbidden_text: str | None = None,
    require_unseen_hole: bool = False,
) -> list[str]:
    clean = text.replace("[[", "").replace("]]", "")
    if len(clean) < (context_chars * 2) + span_chars + 8:
        raise ValueError("text is too small for benchmark case construction")
    rng = random.Random(seed)
    out: list[str] = []
    attempts = 0
    while len(out) < cases and attempts < cases * 200:
        attempts += 1
        start = rng.randint(context_chars, len(clean) - context_chars - span_chars - 1)
        hole = clean[start : start + span_chars]
        if "\n" in hole or not hole.strip():
            continue
        if require_unseen_hole and forbidden_text is not None and hole in forbidden_text:
            continue
        before = clean[start - context_chars : start]
        after = clean[start + span_chars : start + span_chars + context_chars]
        marked = f"{before}[[{hole}]]{after}"
        out.append(marked)
    if len(out) < cases:
        raise ValueError(f"only built {len(out)} benchmark cases from requested {cases}")
    return out


@torch.no_grad()
def masked_eval(
    *,
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    payload: dict[str, Any],
    text: str,
    batches: int,
    batch_size: int,
    mask_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    config = payload["train_config"]
    stream = ByteStream(text, tokenizer, seq_len=payload["model_config"]["seq_len"], split="val", seed=seed)
    losses: list[float] = []
    accs: list[float] = []
    for _ in range(batches):
        clean = stream.sample(batch_size or config["batch_size"], device)
        t = torch.full((clean.shape[0],), mask_rate, device=device)
        corrupted, mask, rates = corrupt_batch(
            clean,
            tokenizer,
            t=t,
            min_mask_rate=config["min_mask_rate"],
            max_mask_rate=config["max_mask_rate"],
            span_prob=0.0,
            max_span_fraction=0.0,
        )
        mask_fraction = mask.float().mean(dim=1)
        logits = restrict_logits_to_ids(
            model(corrupted, rates, corruption_mode=0, mask_fraction=mask_fraction),
            payload.get("sample_token_ids", list(range(model.config.vocab_size))),
        )
        losses.append(float(masked_cross_entropy(logits, clean, mask).item()))
        accs.append(masked_accuracy(logits, clean, mask))
    return {
        "loss": sum(losses) / len(losses),
        "masked_accuracy": sum(accs) / len(accs),
        "mask_rate": mask_rate,
        "batches": batches,
        "batch_size": batch_size,
    }


@torch.no_grad()
def guide_only_case(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide,
    strategy: str,
) -> dict[str, Any]:
    example = parse_marked_infill(marked_text, tokenizer)
    tokens = example.tokens.clone().unsqueeze(0)
    for pos in range(example.hole_start, example.hole_end):
        if strategy == "unigram":
            logits = guide.unigram_log_probs.view(1, 1, -1)
            next_id = int(logits[0, 0].argmax().item())
        elif strategy == "bridge":
            logits = guide.logits(tokens, tokenizer)
            next_id = int(logits[0, pos].argmax().item())
        else:
            raise ValueError(f"unknown guide-only strategy: {strategy}")
        if next_id in {tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id}:
            next_id = tokenizer.byte_offset + ord(" ")
        tokens[0, pos] = next_id
    repaired = tokens[0]
    pred = repaired[example.hole_start : example.hole_end]
    target = example.target[example.hole_start : example.hole_end]
    byte_accuracy = float((pred == target).float().mean().item()) if target.numel() else 0.0
    return {
        "marked_text": marked_text,
        "target_hole": example.hole,
        "predicted_hole": tokenizer.decode(pred),
        "hole_length_bytes": example.hole_length,
        "byte_accuracy": byte_accuracy,
        "exact": bool(torch.equal(pred, target)),
        "frozen_context_unchanged": bool((repaired[example.frozen] == example.target[example.frozen]).all().item()),
        "strategy": strategy,
    }


@torch.no_grad()
def infill_case(
    *,
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide | None,
    guidance: float,
    steps: int,
    top_k: int,
    temperature: float,
    schedule: str,
    seed: int,
    candidates: int = 1,
) -> dict[str, Any]:
    example = parse_marked_infill(marked_text, tokenizer)
    if example.tokens.numel() > model.config.seq_len:
        raise ValueError(f"case has {example.tokens.numel()} tokens; checkpoint seq_len is {model.config.seq_len}")
    started = time.perf_counter()
    best_score = float("-inf")
    best_repaired: torch.Tensor | None = None
    best_trace: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for offset in range(max(1, candidates)):
        repaired, trace = denoise_ids(
            model,
            tokenizer,
            initial_tokens=example.tokens,
            frozen=example.frozen,
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            remask=0.0,
            guide=guide,
            guidance=guidance,
            schedule=schedule,
            seed=seed + offset,
            return_trace=True,
            max_reveal_per_step=1,
            corruption_mode=3,
        )
        assert isinstance(repaired, torch.Tensor)
        pred = repaired[example.hole_start : example.hole_end]
        target = example.target[example.hole_start : example.hole_end]
        score = score_repair(
            model=model,
            tokenizer=tokenizer,
            repaired_ids=repaired,
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
                "predicted_hole": tokenizer.decode(pred),
                "byte_accuracy": float((pred == target).float().mean().item()) if target.numel() else 0.0,
                "exact": bool(torch.equal(pred, target)),
            }
        )
        if score > best_score:
            best_score = score
            best_repaired = repaired
            best_trace = trace
    if best_repaired is None:
        raise RuntimeError("no benchmark infill candidates were produced")
    repaired = best_repaired
    pred = repaired[example.hole_start : example.hole_end]
    target = example.target[example.hole_start : example.hole_end]
    byte_accuracy = float((pred == target).float().mean().item()) if target.numel() else 0.0
    return {
        "marked_text": marked_text,
        "target_hole": example.hole,
        "predicted_hole": tokenizer.decode(pred),
        "hole_length_bytes": example.hole_length,
        "byte_accuracy": byte_accuracy,
        "exact": bool(torch.equal(pred, target)),
        "frozen_context_unchanged": bool((repaired[example.frozen] == example.target[example.frozen]).all().item()),
        "steps_used": len(best_trace),
        "candidates": max(1, candidates),
        "selected_candidate_score": best_score,
        "candidate_summaries": candidate_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize_infill(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"cases": 0, "byte_accuracy": 0.0, "exact_match_rate": 0.0, "pass_at_k_exact": 0.0}
    return {
        "cases": len(rows),
        "byte_accuracy": sum(float(row["byte_accuracy"]) for row in rows) / len(rows),
        "exact_match_rate": sum(1.0 for row in rows if row["exact"]) / len(rows),
        "pass_at_k_exact": sum(
            1.0
            for row in rows
            if (
                any(bool(candidate.get("exact")) for candidate in row.get("candidate_summaries", []))
                if row.get("candidate_summaries")
                else bool(row["exact"])
            )
        )
        / len(rows),
        "frozen_context_ok": all(bool(row["frozen_context_unchanged"]) for row in rows),
    }


def model_quality_label(masked_acc: float, unguided_infill: float, guided_infill: float) -> str:
    if masked_acc >= 0.45 and unguided_infill >= 0.25 and guided_infill >= 0.55:
        return "strong_laptop_checkpoint"
    if masked_acc >= 0.25 and guided_infill >= 0.35:
        return "promising_small_checkpoint"
    if masked_acc >= 0.12:
        return "mechanism_checkpoint"
    return "undertrained"


@torch.no_grad()
def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = choose_device(args.device)
    model, tokenizer, payload = load_checkpoint(args.checkpoint, device=device)
    text = load_text(args.data)
    train_text, val_text = split_text(text, args.val_fraction)
    cases = make_marked_cases(
        val_text,
        cases=args.cases,
        span_chars=args.span_chars,
        context_chars=args.context_chars,
        seed=args.seed,
        forbidden_text=train_text,
        require_unseen_hole=args.require_unseen_hole,
    )
    masked = masked_eval(
        model=model,
        tokenizer=tokenizer,
        payload=payload,
        text=text,
        batches=args.batches,
        batch_size=args.batch_size or payload["train_config"]["batch_size"],
        mask_rate=args.mask_rate,
        seed=args.seed,
        device=device,
    )
    guide = BigramGuide.from_text(train_text, tokenizer).to_device(device)
    unigram_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    unguided_rows: list[dict[str, Any]] = []
    guided_rows: list[dict[str, Any]] = []
    for index, marked in enumerate(cases):
        unigram_rows.append(guide_only_case(tokenizer=tokenizer, marked_text=marked, guide=guide, strategy="unigram"))
        bridge = guide_only_case(tokenizer=tokenizer, marked_text=marked, guide=guide, strategy="bridge")
        bridge["target_hole_seen_in_train_split"] = bridge["target_hole"] in train_text
        bridge_rows.append(bridge)
        unguided_rows.append(
            infill_case(
                model=model,
                tokenizer=tokenizer,
                marked_text=marked,
                guide=None,
                guidance=0.0,
                steps=args.steps,
                top_k=args.top_k,
                temperature=args.temperature,
                schedule=args.schedule,
                seed=args.seed + index,
                candidates=args.candidates,
            )
        )
        guided = infill_case(
            model=model,
            tokenizer=tokenizer,
            marked_text=marked,
            guide=guide,
            guidance=args.guidance,
            steps=args.steps,
            top_k=args.top_k,
            temperature=args.temperature,
            schedule=args.schedule,
            seed=args.seed + index,
            candidates=args.candidates,
        )
        guided["target_hole_seen_in_train_split"] = guided["target_hole"] in train_text
        guided_rows.append(guided)
    unguided_summary = summarize_infill(unguided_rows)
    unigram_summary = summarize_infill(unigram_rows)
    bridge_summary = summarize_infill(bridge_rows)
    guided_summary = summarize_infill(guided_rows)
    bridge_lift = float(guided_summary["byte_accuracy"]) - float(bridge_summary["byte_accuracy"])
    label = model_quality_label(
        float(masked["masked_accuracy"]),
        float(unguided_summary["byte_accuracy"]),
        float(guided_summary["byte_accuracy"]),
    )
    return {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "loaded_state": payload.get("loaded_state"),
        "state_migration": payload.get("state_migration"),
        "parameters": int(payload.get("metrics", {}).get("parameters", 0)),
        "data": str(args.data),
        "data_bytes": len(text.encode("utf-8")),
        "val_fraction": args.val_fraction,
        "train_split_sha256": sha256_text(train_text),
        "validation_split_sha256": sha256_text(val_text),
        "guide_scope": "training_split_only",
        "case_filter": {
            "require_unseen_hole": bool(args.require_unseen_hole),
            "candidate_count": int(args.candidates),
        },
        "masked_eval": masked,
        "infill": {
            "case_source": "validation_split",
            "span_chars": args.span_chars,
            "context_chars": args.context_chars,
            "unigram_baseline": {"summary": unigram_summary, "cases": unigram_rows},
            "bridge_only_baseline": {"summary": bridge_summary, "cases": bridge_rows},
            "unguided": {"summary": unguided_summary, "cases": unguided_rows},
            "bridge_guided": {"summary": guided_summary, "cases": guided_rows},
            "bridge_guided_minus_bridge_only_byte_accuracy": bridge_lift,
        },
        "quality_label": label,
        "ten_out_of_ten_gate": {
            "artifact": "pass_if_tests_verifier_docs_and_benchmarks_pass",
            "actual_model": "requires strong_laptop_checkpoint or better; otherwise do not claim 10/10 model quality",
            "public_wow": "requires visible infill trace plus non-leaky heldout benchmark",
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a harsher HelixDiff benchmark with held-out infill checks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--mask-rate", type=float, default=0.5)
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--span-chars", type=int, default=6)
    parser.add_argument("--context-chars", type=int, default=36)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--require-unseen-hole", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--schedule", choices=["entropy", "ribbon"], default="ribbon")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    report = benchmark(args)
    print(json.dumps(report, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
