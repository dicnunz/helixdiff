from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .tokenizer import ByteTokenizer


def mask_rate_from_time(
    t: torch.Tensor,
    *,
    min_rate: float = 0.05,
    max_rate: float = 0.95,
    curve: float = 1.35,
) -> torch.Tensor:
    """Map diffusion time in [0, 1] to absorbing-mask probability."""

    t = torch.clamp(t, 0.0, 1.0)
    return min_rate + (max_rate - min_rate) * torch.pow(t, curve)


def corrupt_batch(
    tokens: torch.Tensor,
    tokenizer: ByteTokenizer,
    *,
    t: torch.Tensor | None = None,
    min_mask_rate: float = 0.05,
    max_mask_rate: float = 0.95,
    span_prob: float = 0.35,
    max_span_fraction: float = 0.18,
    ribbon_prob: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply absorbing-mask corruption plus optional span and suffix shocks."""

    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, seq]")
    batch, seq_len = tokens.shape
    device = tokens.device
    if t is None:
        t = torch.rand(batch, device=device)
    rates = mask_rate_from_time(t, min_rate=min_mask_rate, max_rate=max_mask_rate)
    valid = (
        (tokens != tokenizer.pad_token_id)
        & (tokens != tokenizer.bos_token_id)
        & (tokens != tokenizer.eos_token_id)
    )
    base = torch.rand(tokens.shape, device=device) < rates[:, None]
    mask = base & valid

    if span_prob > 0 and max_span_fraction > 0:
        for row in range(batch):
            if torch.rand((), device=device).item() >= span_prob:
                continue
            valid_positions = torch.nonzero(valid[row], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue
            max_span = max(1, int(math.ceil(seq_len * max_span_fraction * float(rates[row].item()))))
            span_len = int(torch.randint(1, max_span + 1, (), device=device).item())
            start_index = int(torch.randint(0, valid_positions.numel(), (), device=device).item())
            start = int(valid_positions[start_index].item())
            end = min(seq_len, start + span_len)
            mask[row, start:end] = valid[row, start:end]

    if ribbon_prob > 0:
        for row in range(batch):
            if torch.rand((), device=device).item() >= ribbon_prob:
                continue
            valid_positions = torch.nonzero(valid[row], as_tuple=False).flatten()
            if valid_positions.numel() <= 1:
                continue
            start_index = int(torch.randint(1, valid_positions.numel(), (), device=device).item())
            start = int(valid_positions[start_index].item())
            mask[row, start:] = mask[row, start:] | valid[row, start:]

    for row in range(batch):
        if not bool(mask[row].any().item()):
            choices = torch.nonzero(valid[row], as_tuple=False).flatten()
            if choices.numel() > 0:
                pick = choices[torch.randint(0, choices.numel(), (), device=device)]
                mask[row, pick] = True

    corrupted = tokens.clone()
    corrupted[mask] = tokenizer.mask_token_id
    return corrupted, mask, rates


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() == 0:
        raise ValueError("masked_cross_entropy received an empty mask")
    flat_logits = logits[mask]
    flat_targets = targets[mask]
    return F.cross_entropy(flat_logits, flat_targets)


def restrict_logits_to_ids(logits: torch.Tensor, allowed_token_ids: list[int]) -> torch.Tensor:
    """Keep training/sampling inside the active corpus vocabulary."""

    allowed = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    allowed[torch.tensor(allowed_token_ids, dtype=torch.long, device=logits.device)] = True
    return logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return float((preds[mask] == targets[mask]).float().mean().item())
