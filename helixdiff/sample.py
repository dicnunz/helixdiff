from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .data import load_text
from .model import DiffusionTransformer, ModelConfig
from .ngram import BigramGuide
from .tokenizer import ByteTokenizer


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[DiffusionTransformer, ByteTokenizer, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    tokenizer = ByteTokenizer()
    config = ModelConfig(**payload["model_config"])
    model = DiffusionTransformer(config, pad_token_id=tokenizer.pad_token_id).to(device)
    model.load_state_dict(payload["model_state"])
    model.allowed_token_ids = payload.get("sample_token_ids")
    model.eval()
    return model, tokenizer, payload


def _filter_logits(
    logits: torch.Tensor,
    tokenizer: ByteTokenizer,
    top_k: int,
    *,
    allowed_token_ids: list[int] | None = None,
    allow_eos: bool = False,
) -> torch.Tensor:
    logits = logits.clone()
    if allowed_token_ids:
        allowed = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        allowed[torch.tensor(allowed_token_ids, dtype=torch.long, device=logits.device)] = True
        logits = logits.masked_fill(~allowed, -torch.inf)
    logits[..., tokenizer.pad_token_id] = -torch.inf
    logits[..., tokenizer.mask_token_id] = -torch.inf
    logits[..., tokenizer.bos_token_id] = -torch.inf
    if not allow_eos:
        logits[..., tokenizer.eos_token_id] = -torch.inf
    if top_k > 0 and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, -torch.inf)
    return logits


def render_tokens_with_masks(
    ids: torch.Tensor | list[int],
    tokenizer: ByteTokenizer,
    *,
    mask_char: str = "~",
) -> str:
    """Render byte tokens while keeping unrepaired masks visible."""
    pieces: list[str] = []
    raw = bytearray()

    def flush_raw() -> None:
        nonlocal raw
        if raw:
            pieces.append(bytes(raw).decode("utf-8", errors="replace"))
            raw = bytearray()

    for token_id in ids:
        token_id = int(token_id)
        if token_id == tokenizer.mask_token_id:
            flush_raw()
            pieces.append(mask_char)
        elif token_id in {tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id}:
            flush_raw()
        elif token_id >= tokenizer.byte_offset:
            raw.append(token_id - tokenizer.byte_offset)
        else:
            flush_raw()
            pieces.append(tokenizer.special_name(token_id))
    flush_raw()
    return "".join(pieces)


@torch.no_grad()
def denoise_ids(
    model: DiffusionTransformer,
    tokenizer: ByteTokenizer,
    *,
    initial_tokens: torch.Tensor,
    frozen: torch.Tensor | None = None,
    steps: int = 48,
    temperature: float = 0.9,
    top_k: int = 64,
    remask: float = 0.05,
    allow_eos: bool = False,
    guide: BigramGuide | None = None,
    guidance: float = 0.0,
    schedule: str = "entropy",
    seed: int = 7,
    return_trace: bool = False,
    trace_preview: bool = False,
    max_reveal_per_step: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, Any]]]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if schedule not in {"entropy", "ribbon"}:
        raise ValueError("schedule must be 'entropy' or 'ribbon'")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    tokens = initial_tokens.to(device=device, dtype=torch.long).clone()
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("initial_tokens must be a 1D tensor or a single-row 2D tensor")
    if tokens.shape[1] > model.config.seq_len:
        raise ValueError(f"initial token length {tokens.shape[1]} exceeds model seq_len {model.config.seq_len}")
    if frozen is None:
        frozen_mask = tokens != tokenizer.mask_token_id
    else:
        frozen_mask = frozen.to(device=device, dtype=torch.bool).clone()
        if frozen_mask.ndim == 1:
            frozen_mask = frozen_mask.unsqueeze(0)
        if frozen_mask.shape != tokens.shape:
            raise ValueError("frozen mask must have the same shape as initial_tokens")
    generated_once = torch.zeros_like(tokens, dtype=torch.bool)
    generated_once[(tokens != tokenizer.mask_token_id) & ~frozen_mask] = True
    trace_rows: list[dict[str, Any]] = []

    for step in range(steps):
        remaining_steps = steps - step
        t = torch.full((1,), remaining_steps / steps, device=device)
        logits = model(tokens, t)
        if guide is not None and guidance > 0:
            logits = logits + guidance * guide.logits(tokens, tokenizer)
        logits = _filter_logits(
            logits / max(temperature, 1e-4),
            tokenizer,
            top_k=top_k,
            allowed_token_ids=getattr(model, "allowed_token_ids", None),
            allow_eos=allow_eos,
        )
        probs = torch.softmax(logits, dim=-1)
        sample = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=generator).view_as(tokens)
        confidence = probs.gather(-1, sample.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * torch.clamp(probs, min=1e-8).log()).sum(dim=-1) / math.log(probs.shape[-1])
        masked = tokens == tokenizer.mask_token_id
        remaining = int(masked.sum().item())
        if remaining == 0:
            break
        masked_entropy = entropy[masked]
        mean_masked_entropy = float(masked_entropy.mean().item()) if masked_entropy.numel() else 0.0
        if remaining_steps == 1:
            reveal_count = remaining
        else:
            certainty = 1.0 - mean_masked_entropy
            reveal_fraction = max(1.0 / remaining_steps, 0.06 + 0.42 * max(0.0, certainty))
            reveal_count = max(1, min(remaining, int(math.ceil(remaining * reveal_fraction))))
        if max_reveal_per_step is not None:
            reveal_count = max(1, min(reveal_count, int(max_reveal_per_step)))
        if schedule == "ribbon":
            masked_flat = torch.nonzero(masked.reshape(-1), as_tuple=False).flatten()
            reveal_flat = masked_flat[:reveal_count]
        else:
            score = confidence - 0.15 * entropy
            score = score.masked_fill(~masked, -torch.inf)
            reveal_flat = torch.topk(score.reshape(-1), reveal_count).indices
        rows = reveal_flat // tokens.shape[1]
        cols = reveal_flat % tokens.shape[1]
        tokens[rows, cols] = sample[rows, cols]
        generated_once[rows, cols] = True
        mean_reveal_confidence = float(confidence[rows, cols].mean().item()) if reveal_count else 0.0

        remask_count = 0
        if remask > 0 and step < steps - 2:
            eligible = generated_once & ~frozen_mask & (tokens != tokenizer.mask_token_id)
            eligible_count = int(eligible.sum().item())
            remask_count = int(eligible_count * remask)
            if remask_count > 0:
                uncertainty = entropy.masked_fill(~eligible, -torch.inf)
                remask_flat = torch.topk(uncertainty.reshape(-1), remask_count).indices
                r_rows = remask_flat // tokens.shape[1]
                r_cols = remask_flat % tokens.shape[1]
                tokens[r_rows, r_cols] = tokenizer.mask_token_id
        if return_trace:
            remaining_after = int((tokens == tokenizer.mask_token_id).sum().item())
            row: dict[str, Any] = {
                "step": step + 1,
                "remaining_before": remaining,
                "revealed": int(reveal_count),
                "remasked": int(remask_count),
                "remaining_after": remaining_after,
                "visible_fraction": round(1.0 - (remaining_after / tokens.numel()), 6),
                "mean_masked_entropy": round(mean_masked_entropy, 6),
                "mean_reveal_confidence": round(mean_reveal_confidence, 6),
            }
            if trace_preview:
                row["preview"] = render_tokens_with_masks(tokens[0].detach().cpu(), tokenizer)
            trace_rows.append(row)

    tokens[tokens == tokenizer.mask_token_id] = tokenizer.eos_token_id
    result = tokens[0].detach().cpu()
    if return_trace:
        return result, trace_rows
    return result


def generate_ids(
    model: DiffusionTransformer,
    tokenizer: ByteTokenizer,
    *,
    prompt: str = "",
    total_tokens: int = 160,
    steps: int = 48,
    temperature: float = 0.9,
    top_k: int = 64,
    remask: float = 0.05,
    allow_eos: bool = False,
    guide: BigramGuide | None = None,
    guidance: float = 0.0,
    schedule: str = "entropy",
    scaffold: bool = False,
    scaffold_remask: float = 0.18,
    seed: int = 7,
) -> torch.Tensor:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    total_tokens = min(max(total_tokens, len(prompt_ids) + 8), model.config.seq_len)
    tokens = torch.full((1, total_tokens), tokenizer.mask_token_id, dtype=torch.long, device=device)
    tokens[0, : len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    if scaffold and guide is not None:
        scaffold_ids = guide.scaffold_ids(
            prompt_ids,
            total_tokens,
            tokenizer,
            seed=seed,
            temperature=temperature,
        )
        tokens[0] = torch.tensor(scaffold_ids, dtype=torch.long, device=device)
        remaskable = torch.zeros_like(tokens, dtype=torch.bool)
        remaskable[:, len(prompt_ids) :] = True
        if scaffold_remask > 0:
            remask_draw = torch.rand(tokens.shape, device=device, generator=generator) < scaffold_remask
            tokens[remaskable & remask_draw] = tokenizer.mask_token_id
    frozen = tokens != tokenizer.mask_token_id
    frozen[:, len(prompt_ids) :] = False
    result = denoise_ids(
        model,
        tokenizer,
        initial_tokens=tokens,
        frozen=frozen,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        remask=remask,
        allow_eos=allow_eos,
        guide=guide,
        guidance=guidance,
        schedule=schedule,
        seed=seed,
    )
    return result if isinstance(result, torch.Tensor) else result[0]


def generate_text(
    model: DiffusionTransformer,
    tokenizer: ByteTokenizer,
    **kwargs,
) -> str:
    return tokenizer.decode(generate_ids(model, tokenizer, **kwargs))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample from a HelixDiff checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--remask", type=float, default=0.05)
    parser.add_argument("--allow-eos", action="store_true")
    parser.add_argument("--guide-data")
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--schedule", choices=["entropy", "ribbon"], default="entropy")
    parser.add_argument("--scaffold", action="store_true")
    parser.add_argument("--scaffold-remask", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    device = choose_device(args.device)
    model, tokenizer, payload = load_checkpoint(args.checkpoint, device=device)
    guide = None
    if args.guide_data and (args.guidance > 0 or args.scaffold):
        guide = BigramGuide.from_text(load_text(args.guide_data), tokenizer).to_device(device)
    start = time.perf_counter()
    text = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        total_tokens=args.tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        remask=args.remask,
        allow_eos=args.allow_eos,
        guide=guide,
        guidance=args.guidance,
        schedule=args.schedule,
        scaffold=args.scaffold,
        scaffold_remask=args.scaffold_remask,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start
    print(text)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "prompt": args.prompt,
                    "tokens": args.tokens,
                    "steps": args.steps,
                    "elapsed_seconds": elapsed,
                    "guide_data": args.guide_data,
                    "guidance": args.guidance,
                    "schedule": args.schedule,
                    "scaffold": args.scaffold,
                    "scaffold_remask": args.scaffold_remask,
                    "checkpoint_step": payload.get("step"),
                    "text": text,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
