from __future__ import annotations

from pathlib import Path

import torch

from .tokenizer import ByteTokenizer


DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "seed_corpus.txt"


def load_text(path: str | Path | None = None) -> str:
    source = Path(path) if path else DEFAULT_DATA
    return source.read_text(encoding="utf-8")


class ByteStream:
    """Random contiguous byte windows for language-model pretraining."""

    def __init__(
        self,
        text: str,
        tokenizer: ByteTokenizer,
        *,
        seq_len: int,
        split: str = "train",
        val_fraction: float = 0.08,
        seed: int = 0,
    ) -> None:
        if seq_len < 8:
            raise ValueError("seq_len must be at least 8")
        encoded = tokenizer.encode(text, add_bos=False, add_eos=False)
        if len(encoded) < seq_len * 4:
            repeat = (seq_len * 4 // max(1, len(encoded))) + 2
            encoded = encoded * repeat
        split_at = max(seq_len + 1, int(len(encoded) * (1.0 - val_fraction)))
        if split == "train":
            ids = encoded[:split_at]
        elif split in {"val", "eval", "validation"}:
            ids = encoded[max(0, split_at - seq_len) :]
        else:
            raise ValueError(f"unknown split: {split}")
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        limit = len(self.tokens) - (self.seq_len - 2)
        if limit <= 1:
            raise ValueError("corpus is too small for requested sequence length")
        starts = torch.randint(0, limit, (batch_size,), generator=self.generator)
        rows = []
        for start in starts.tolist():
            chunk = self.tokens[start : start + self.seq_len - 2]
            row = torch.empty(self.seq_len, dtype=torch.long)
            row[0] = self.tokenizer.bos_token_id
            row[1:-1] = chunk
            row[-1] = self.tokenizer.eos_token_id
            rows.append(row)
        return torch.stack(rows).to(device)

