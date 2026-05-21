from __future__ import annotations

import random
from collections import Counter, defaultdict
from math import log

import torch

from .tokenizer import ByteTokenizer


class BigramGuide:
    """Scratch-built byte n-gram guide for diffusion sampling."""

    def __init__(
        self,
        left_log_probs: torch.Tensor,
        right_log_probs: torch.Tensor,
        unigram_log_probs: torch.Tensor,
        transitions: dict[tuple[int, ...], Counter[int]] | None = None,
        *,
        order: int = 5,
    ) -> None:
        self.left_log_probs = left_log_probs
        self.right_log_probs = right_log_probs
        self.unigram_log_probs = unigram_log_probs
        self.transitions = transitions or {}
        self.order = int(order)

    @classmethod
    def from_text(cls, text: str, tokenizer: ByteTokenizer, *, smoothing: float = 0.25, order: int = 5) -> "BigramGuide":
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        vocab = tokenizer.vocab_size
        left = torch.full((vocab, vocab), smoothing, dtype=torch.float32)
        right = torch.full((vocab, vocab), smoothing, dtype=torch.float32)
        unigram = torch.full((vocab,), smoothing, dtype=torch.float32)
        transitions: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for token_id in ids:
            unigram[token_id] += 1.0
        for a, b in zip(ids, ids[1:]):
            left[a, b] += 1.0
            right[b, a] += 1.0
        max_order = max(2, int(order))
        for i in range(1, len(ids)):
            for size in range(1, max_order):
                start = max(0, i - size)
                context = tuple(ids[start:i])
                if context:
                    transitions[context][ids[i]] += 1
        left = torch.log(left / left.sum(dim=-1, keepdim=True))
        right = torch.log(right / right.sum(dim=-1, keepdim=True))
        unigram = torch.log(unigram / unigram.sum())
        return cls(left, right, unigram, dict(transitions), order=max_order)

    def to_device(self, device: torch.device | str) -> "BigramGuide":
        return BigramGuide(
            self.left_log_probs.to(device),
            self.right_log_probs.to(device),
            self.unigram_log_probs.to(device),
            self.transitions,
            order=self.order,
        )

    def scaffold_ids(
        self,
        prompt_ids: list[int],
        total_tokens: int,
        tokenizer: ByteTokenizer,
        *,
        seed: int,
        temperature: float = 1.0,
    ) -> list[int]:
        rng = random.Random(seed)
        ids = list(prompt_ids[:total_tokens])
        fallback_ids = torch.softmax(self.unigram_log_probs.cpu(), dim=0).tolist()
        blocked = {tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id}
        while len(ids) < total_tokens:
            counts = None
            max_context = min(self.order - 1, len(ids))
            for size in range(max_context, 0, -1):
                counts = self.transitions.get(tuple(ids[-size:]))
                if counts:
                    break
            if counts:
                candidates = [token_id for token_id in counts if token_id not in blocked]
                weights = [float(counts[token_id]) ** (1.0 / max(temperature, 1e-4)) for token_id in candidates]
            else:
                candidates = [token_id for token_id, weight in enumerate(fallback_ids) if weight > 0 and token_id not in blocked]
                weights = [fallback_ids[token_id] ** (1.0 / max(temperature, 1e-4)) for token_id in candidates]
            ids.append(rng.choices(candidates, weights=weights, k=1)[0])
        return ids

    def logits(self, tokens: torch.Tensor, tokenizer: ByteTokenizer) -> torch.Tensor:
        batch, seq_len = tokens.shape
        vocab = self.unigram_log_probs.shape[0]
        out = self.unigram_log_probs.view(1, 1, vocab).expand(batch, seq_len, vocab).clone()
        known = (
            (tokens != tokenizer.mask_token_id)
            & (tokens != tokenizer.pad_token_id)
            & (tokens != tokenizer.bos_token_id)
        )
        for pos in range(seq_len):
            if pos > 0:
                left_ids = tokens[:, pos - 1]
                left_known = known[:, pos - 1]
                if bool(left_known.any().item()):
                    out[left_known, pos, :] = out[left_known, pos, :] + self.left_log_probs[left_ids[left_known]]
            if pos + 1 < seq_len:
                right_ids = tokens[:, pos + 1]
                right_known = known[:, pos + 1]
                if bool(right_known.any().item()):
                    out[right_known, pos, :] = out[right_known, pos, :] + self.right_log_probs[right_ids[right_known]]
            max_context = min(self.order - 1, pos)
            if max_context <= 0:
                continue
            for row in range(batch):
                for size in range(max_context, 0, -1):
                    start = pos - size
                    if not bool(known[row, start:pos].all().item()):
                        continue
                    context = tuple(int(token_id) for token_id in tokens[row, start:pos].tolist())
                    counts = self.transitions.get(context)
                    if not counts:
                        continue
                    total = float(sum(counts.values()))
                    bonus = out.new_full((vocab,), log(1e-4 / vocab))
                    denom = total + (1e-4 * vocab)
                    for token_id, count in counts.items():
                        bonus[int(token_id)] = log((float(count) + 1e-4) / denom)
                    out[row, pos, :] = out[row, pos, :] + bonus
                    break
        return out
