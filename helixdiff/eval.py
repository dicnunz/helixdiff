from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import ByteStream, load_text
from .diffusion import corrupt_batch, masked_accuracy, masked_cross_entropy, restrict_logits_to_ids
from .ngram import BigramGuide
from .sample import choose_device, generate_text, load_checkpoint


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    device = choose_device(args.device)
    model, tokenizer, payload = load_checkpoint(args.checkpoint, device=device)
    config = payload["train_config"]
    text = load_text(args.data)
    stream = ByteStream(text, tokenizer, seq_len=payload["model_config"]["seq_len"], split="val", seed=args.seed)
    losses: list[float] = []
    accs: list[float] = []
    for _ in range(args.batches):
        clean = stream.sample(args.batch_size or config["batch_size"], device)
        t = torch.full((clean.shape[0],), args.mask_rate, device=device)
        corrupted, mask, rates = corrupt_batch(
            clean,
            tokenizer,
            t=t,
            min_mask_rate=config["min_mask_rate"],
            max_mask_rate=config["max_mask_rate"],
            span_prob=0.0,
            max_span_fraction=0.0,
        )
        logits = restrict_logits_to_ids(model(corrupted, rates), payload.get("sample_token_ids", list(range(model.config.vocab_size))))
        losses.append(float(masked_cross_entropy(logits, clean, mask).item()))
        accs.append(masked_accuracy(logits, clean, mask))
    start = time.perf_counter()
    guide = None
    if args.guide_data:
        guide = BigramGuide.from_text(load_text(args.guide_data), tokenizer).to_device(device)
    smoke = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        total_tokens=args.tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        remask=args.remask,
        guide=guide,
        guidance=args.guidance,
        schedule=args.schedule,
        scaffold=args.scaffold,
        scaffold_remask=args.scaffold_remask,
        seed=args.seed + 99,
    )
    elapsed = time.perf_counter() - start
    return {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "loss": sum(losses) / len(losses),
        "masked_accuracy": sum(accs) / len(accs),
        "mask_rate": args.mask_rate,
        "batches": args.batches,
        "sample_elapsed_seconds": elapsed,
        "sample_bytes_per_second": len(smoke.encode("utf-8")) / max(elapsed, 1e-9),
        "sample_schedule": args.schedule,
        "sample_scaffold": args.scaffold,
        "sample": smoke,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a HelixDiff checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--mask-rate", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prompt", default="The denoiser")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--remask", type=float, default=0.06)
    parser.add_argument("--guide-data")
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--schedule", choices=["entropy", "ribbon"], default="entropy")
    parser.add_argument("--scaffold", action="store_true")
    parser.add_argument("--scaffold-remask", type=float, default=0.18)
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    report = evaluate(args)
    print(json.dumps(report, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
