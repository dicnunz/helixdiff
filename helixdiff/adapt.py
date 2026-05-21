from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import AdamW

from .diffusion import masked_accuracy, masked_cross_entropy, restrict_logits_to_ids
from .model import DiffusionTransformer
from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class VisibleAdaptConfig:
    steps: int = 0
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    span_min: int = 3
    span_max: int = 12
    train_scope: str = "last_block"
    seed: int = 7


def visible_context_text(example: Any) -> str:
    return example.before + example.after


def visible_context_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def set_adaptation_train_scope(model: DiffusionTransformer, scope: str) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if scope == "head":
        modules = [model.norm, model.lm_head]
    elif scope == "last_block":
        modules = [model.blocks[-1], model.norm, model.lm_head]
    elif scope == "all":
        modules = [model]
    else:
        raise ValueError("train_scope must be one of: head, last_block, all")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def make_visible_suture_batch(
    *,
    visible_text: str,
    tokenizer: ByteTokenizer,
    seq_len: int,
    batch_size: int,
    span_min: int,
    span_max: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not visible_text.strip():
        raise ValueError("visible context is empty; cannot adapt")
    clean_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    span_min = max(1, int(span_min))
    span_max = max(span_min, int(span_max))
    for _ in range(batch_size):
        encoded = tokenizer.encode(visible_text, add_bos=True, add_eos=True)
        if len(encoded) > seq_len:
            max_start = max(0, len(encoded) - seq_len)
            crop_start = int(torch.randint(0, max_start + 1, (), generator=generator).item())
            encoded = encoded[crop_start : crop_start + seq_len]
            if encoded[0] != tokenizer.bos_token_id:
                encoded[0] = tokenizer.bos_token_id
            if encoded[-1] != tokenizer.eos_token_id:
                encoded[-1] = tokenizer.eos_token_id
        if len(encoded) < seq_len:
            encoded = encoded + [tokenizer.pad_token_id] * (seq_len - len(encoded))
        clean = torch.tensor(encoded[:seq_len], dtype=torch.long, device=device)
        valid = (
            (clean != tokenizer.pad_token_id)
            & (clean != tokenizer.bos_token_id)
            & (clean != tokenizer.eos_token_id)
        )
        valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            raise ValueError("visible context has no repairable byte positions")
        span_len = min(
            int(torch.randint(span_min, span_max + 1, (), generator=generator).item()),
            int(valid_positions.numel()),
        )
        start_slot = int(torch.randint(0, int(valid_positions.numel() - span_len + 1), (), generator=generator).item())
        start = int(valid_positions[start_slot].item())
        end = int(valid_positions[start_slot + span_len - 1].item()) + 1
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[start:end] = valid[start:end]
        clean_rows.append(clean)
        mask_rows.append(mask)
    clean_batch = torch.stack(clean_rows, dim=0)
    mask_batch = torch.stack(mask_rows, dim=0)
    corrupted = clean_batch.clone()
    corrupted[mask_batch] = tokenizer.mask_token_id
    return clean_batch, corrupted, mask_batch


def adapt_model_to_visible_context(
    *,
    model: DiffusionTransformer,
    tokenizer: ByteTokenizer,
    example: Any,
    config: VisibleAdaptConfig,
) -> tuple[DiffusionTransformer, dict[str, Any]]:
    if config.steps <= 0:
        return model, {
            "enabled": False,
            "steps": 0,
            "visible_context_sha256": visible_context_hash(visible_context_text(example)),
        }
    device = next(model.parameters()).device
    adapted = copy.deepcopy(model).to(device)
    adapted.allowed_token_ids = getattr(model, "allowed_token_ids", None)
    visible_text = visible_context_text(example)
    target_seen = bool(example.hole and example.hole in visible_text)
    trainable_names = set_adaptation_train_scope(adapted, config.train_scope)
    parameters = [parameter for parameter in adapted.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("visible adaptation selected no trainable parameters")
    optimizer = AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    losses: list[float] = []
    accs: list[float] = []
    started = time.perf_counter()
    adapted.train()
    for _ in range(config.steps):
        clean, corrupted, mask = make_visible_suture_batch(
            visible_text=visible_text,
            tokenizer=tokenizer,
            seq_len=adapted.config.seq_len,
            batch_size=config.batch_size,
            span_min=config.span_min,
            span_max=config.span_max,
            device=device,
            generator=generator,
        )
        rates = mask.float().mean(dim=1).clamp_min(1.0 / adapted.config.seq_len)
        logits = restrict_logits_to_ids(
            adapted(corrupted, rates, corruption_mode=3, mask_fraction=rates),
            getattr(adapted, "allowed_token_ids", None) or list(range(adapted.config.vocab_size)),
        )
        loss = masked_cross_entropy(logits, clean, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        accs.append(masked_accuracy(logits.detach(), clean, mask))
    adapted.eval()
    elapsed = time.perf_counter() - started
    return adapted, {
        "enabled": True,
        "mechanism": "visible_context_suture_tta",
        "steps": int(config.steps),
        "batch_size": int(config.batch_size),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "span_min": int(config.span_min),
        "span_max": int(config.span_max),
        "train_scope": config.train_scope,
        "seed": int(config.seed),
        "visible_context_sha256": visible_context_hash(visible_text),
        "visible_context_bytes": len(visible_text.encode("utf-8")),
        "hidden_target_seen_in_visible_context": target_seen,
        "trainable_parameter_names": trainable_names,
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_masked_accuracy": sum(accs) / len(accs) if accs else 0.0,
        "elapsed_seconds": elapsed,
    }
