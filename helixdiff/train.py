from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import trange

from .data import ByteStream, load_text
from .diffusion import corrupt_batch, masked_accuracy, masked_cross_entropy, restrict_logits_to_ids, suture_boundary_weights
from .model import DiffusionTransformer, ModelConfig, count_parameters
from .ngram import BigramGuide
from .sample import choose_device, generate_text
from .tokenizer import ByteTokenizer


@dataclass
class TrainConfig:
    name: str = "helixdiff"
    seq_len: int = 128
    dim: int = 128
    layers: int = 4
    heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 8e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.05
    warmup_steps: int = 100
    min_mask_rate: float = 0.05
    max_mask_rate: float = 0.95
    span_prob: float = 0.35
    max_span_fraction: float = 0.18
    ribbon_prob: float = 0.25
    suture_prob: float = 0.0
    suture_min_span: int = 3
    suture_max_span: int = 12
    suture_boundary_weight: float = 1.0


def load_config(path: str | Path | None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = set(TrainConfig.__dataclass_fields__)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return TrainConfig(**data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_for_step(base_lr: float, step: int, warmup: int, total: int, min_lr_ratio: float) -> float:
    if step <= warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def save_checkpoint(
    path: str | Path,
    model: DiffusionTransformer,
    config: TrainConfig,
    tokenizer: ByteTokenizer,
    *,
    step: int,
    metrics: dict,
    sample_token_ids: list[int],
    optimizer: torch.optim.Optimizer | None = None,
    resumed_from: str | None = None,
    ema_model_state: dict[str, torch.Tensor] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "helixdiff-checkpoint-v1",
        "scratch_only": True,
        "pretrained_weights": False,
        "hosted_model_calls": False,
        "step": step,
        "train_config": config.__dict__,
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_metadata(),
        "sample_token_ids": sample_token_ids,
        "model_state": model.state_dict(),
        "ema_model_state": ema_model_state,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "resumed_from": resumed_from,
        "novel_mechanisms": [
            "span_shock_corruption",
            "ribbon_suffix_corruption",
            "entropy_clock_sampling",
            "ribbon_decode_sampling",
            "confidence_remasking",
            "corpus_ngram_scaffold_sampling",
            "suture_trace_infill",
            "high_order_bridge_guidance",
            "boundary_pinned_suture_corruption",
            "clock_conditioned_denoiser",
            "corruption_mode_conditioned_denoiser",
        ],
    }
    torch.save(payload, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train HelixDiff from scratch.")
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--data")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--checkpoint", default="checkpoints/helixdiff_tiny.pt")
    parser.add_argument("--sample-out", default="samples/latest.txt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--prompt", default="Language is rebuilt by")
    parser.add_argument("--resume")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    config = load_config(args.config)
    tokenizer = ByteTokenizer()
    device = choose_device(args.device)
    text = load_text(args.data)
    sample_token_ids = sorted(set(tokenizer.encode(text, add_bos=False, add_eos=False) + [tokenizer.eos_token_id]))
    stream = ByteStream(text, tokenizer, seq_len=config.seq_len, split="train", seed=args.seed)
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        seq_len=config.seq_len,
        dim=config.dim,
        layers=config.layers,
        heads=config.heads,
        ff_mult=config.ff_mult,
        dropout=config.dropout,
    )
    model = DiffusionTransformer(model_config, pad_token_id=tokenizer.pad_token_id).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    start_step = 1
    resumed_from = None
    ema_state: dict[str, torch.Tensor] | None = None
    if args.resume:
        resume_path = Path(args.resume)
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        saved_model_config = dict(payload.get("model_config", {}))
        current_model_config = model.config.to_dict()
        for migrated_key in ("noise_buckets", "condition_modes"):
            saved_model_config.setdefault(migrated_key, current_model_config[migrated_key])
        if saved_model_config != current_model_config:
            raise ValueError(f"resume checkpoint model_config does not match requested config: {saved_model_config}")
        missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
        allowed_missing = tuple(name for name in missing if name.startswith(("noise_emb.", "mode_emb.")))
        if unexpected or len(allowed_missing) != len(missing):
            raise RuntimeError(f"resume checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
        if any(name.startswith("noise_emb.") for name in missing):
            model.noise_emb.weight.data.zero_()
        if any(name.startswith("mode_emb.") for name in missing):
            model.mode_emb.weight.data.zero_()
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        saved_ema = payload.get("ema_model_state")
        if saved_ema:
            ema_state = {key: value.to(device=device).clone() for key, value in saved_ema.items()}
        start_step = int(payload.get("step", 0)) + 1
        resumed_from = str(resume_path)
        if start_step > args.steps:
            raise ValueError(f"resume checkpoint is already at step {start_step - 1}; requested final step is {args.steps}")
    if args.ema_decay > 0 and ema_state is None:
        ema_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    start = time.perf_counter()
    metrics: dict[str, float | int | str] = {
        "loss": float("nan"),
        "masked_accuracy": 0.0,
        "parameters": count_parameters(model),
        "device": str(device),
        "ema_decay": float(args.ema_decay),
        "resumed_from": resumed_from or "",
    }

    bar = trange(start_step, args.steps + 1, desc="train", dynamic_ncols=True)
    for step in bar:
        model.train()
        lr = lr_for_step(config.learning_rate, step, config.warmup_steps, args.steps, config.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        clean = stream.sample(config.batch_size, device)
        t = torch.rand(config.batch_size, device=device)
        corrupted, mask, rates, modes = corrupt_batch(
            clean,
            tokenizer,
            t=t,
            min_mask_rate=config.min_mask_rate,
            max_mask_rate=config.max_mask_rate,
            span_prob=config.span_prob,
            max_span_fraction=config.max_span_fraction,
            ribbon_prob=config.ribbon_prob,
            suture_prob=config.suture_prob,
            suture_min_span=config.suture_min_span,
            suture_max_span=config.suture_max_span,
            return_mode=True,
        )
        mask_fraction = mask.float().mean(dim=1)
        logits = restrict_logits_to_ids(
            model(corrupted, rates, corruption_mode=modes, mask_fraction=mask_fraction),
            sample_token_ids,
        )
        loss_weights = None
        if config.suture_boundary_weight > 1.0:
            loss_weights = suture_boundary_weights(mask, modes, boundary_weight=config.suture_boundary_weight)
        loss = masked_cross_entropy(logits, clean, mask, weights=loss_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema_state is not None:
            decay = min(args.ema_decay, (1.0 + step) / (10.0 + step))
            current_state = model.state_dict()
            for key, value in current_state.items():
                if torch.is_floating_point(value):
                    ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
                else:
                    ema_state[key].copy_(value)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            acc = masked_accuracy(logits.detach(), clean, mask)
            metrics.update(
                {
                    "loss": float(loss.item()),
                    "masked_accuracy": acc,
                    "step": step,
                    "learning_rate": lr,
                    "mean_mask_rate": float(rates.mean().item()),
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
            bar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc:.3f}", mask=f"{rates.mean().item():.2f}")

    save_checkpoint(
        args.checkpoint,
        model,
        config,
        tokenizer,
        step=args.steps,
        metrics=metrics,
        sample_token_ids=sample_token_ids,
        optimizer=optimizer,
        resumed_from=resumed_from,
        ema_model_state=ema_state,
    )
    model.eval()
    if ema_state is not None:
        model.load_state_dict(ema_state)
    sample = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        total_tokens=min(config.seq_len, 180),
        steps=48,
        temperature=0.85,
        top_k=32,
        remask=0.0,
        guide=BigramGuide.from_text(text, tokenizer).to_device(device),
        guidance=0.3,
        schedule="ribbon",
        scaffold=True,
        scaffold_remask=0.03,
        seed=args.seed + 1,
    )
    if args.sample_out:
        sample_path = Path(args.sample_out)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(sample, encoding="utf-8")
    print(json.dumps({"checkpoint": args.checkpoint, "sample_out": args.sample_out, **metrics}, indent=2))


if __name__ == "__main__":
    main()
