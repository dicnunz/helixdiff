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
    suture_prob: float = 0.0,
    suture_min_span: int = 3,
    suture_max_span: int = 12,
    return_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    modes = torch.zeros(batch, dtype=torch.long, device=device)

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
            modes[row] = 1

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
            modes[row] = 2

    if suture_prob > 0:
        for row in range(batch):
            if torch.rand((), device=device).item() >= suture_prob:
                continue
            valid_positions = torch.nonzero(valid[row], as_tuple=False).flatten()
            if valid_positions.numel() <= 4:
                continue
            max_span = min(max(int(suture_min_span), int(suture_max_span)), max(1, valid_positions.numel() - 2))
            min_span = min(max(1, int(suture_min_span)), max_span)
            span_len = int(torch.randint(min_span, max_span + 1, (), device=device).item())
            start_slot = int(torch.randint(1, valid_positions.numel() - span_len, (), device=device).item())
            start = int(valid_positions[start_slot].item())
            end = int(valid_positions[start_slot + span_len - 1].item()) + 1
            mask[row] = False
            mask[row, start:end] = valid[row, start:end]
            modes[row] = 3

    for row in range(batch):
        if not bool(mask[row].any().item()):
            choices = torch.nonzero(valid[row], as_tuple=False).flatten()
            if choices.numel() > 0:
                pick = choices[torch.randint(0, choices.numel(), (), device=device)]
                mask[row, pick] = True

    corrupted = tokens.clone()
    corrupted[mask] = tokenizer.mask_token_id
    if return_mode:
        return corrupted, mask, rates, modes
    return corrupted, mask, rates


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask.sum().item() == 0:
        raise ValueError("masked_cross_entropy received an empty mask")
    flat_logits = logits[mask]
    flat_targets = targets[mask]
    losses = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    if weights is None:
        return losses.mean()
    flat_weights = weights[mask].to(dtype=losses.dtype)
    return (losses * flat_weights).sum() / flat_weights.sum().clamp_min(1e-8)


def suture_boundary_weights(mask: torch.Tensor, modes: torch.Tensor, *, boundary_weight: float = 2.0) -> torch.Tensor:
    """Upweight the first and last masked tokens of bounded suture repairs."""

    weights = torch.ones(mask.shape, dtype=torch.float32, device=mask.device)
    if boundary_weight <= 1.0:
        return weights
    for row in range(mask.shape[0]):
        if int(modes[row].item()) != 3:
            continue
        masked_positions = torch.nonzero(mask[row], as_tuple=False).flatten()
        if masked_positions.numel() == 0:
            continue
        weights[row, masked_positions[0]] = float(boundary_weight)
        weights[row, masked_positions[-1]] = float(boundary_weight)
    return weights


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
