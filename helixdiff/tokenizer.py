from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ByteTokenizer:
    """Tiny UTF-8 byte tokenizer with no external files."""

    pad_token_id: int = 0
    mask_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    byte_offset: int = 4

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + 256

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        ids.extend(byte + self.byte_offset for byte in text.encode("utf-8", errors="replace"))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], *, strip_special: bool = True) -> str:
        raw = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if token_id >= self.byte_offset:
                raw.append(token_id - self.byte_offset)
            elif not strip_special:
                raw.extend(self.special_name(token_id).encode("utf-8"))
        return bytes(raw).decode("utf-8", errors="replace")

    def special_name(self, token_id: int) -> str:
        names = {
            self.pad_token_id: "<pad>",
            self.mask_token_id: "<mask>",
            self.bos_token_id: "<bos>",
            self.eos_token_id: "<eos>",
        }
        return names.get(int(token_id), f"<special:{int(token_id)}>")

    def is_special(self, token_id: int) -> bool:
        return int(token_id) < self.byte_offset

    def to_metadata(self) -> dict[str, int | str]:
        return {
            "type": "ByteTokenizer",
            "version": "1",
            "pad_token_id": self.pad_token_id,
            "mask_token_id": self.mask_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "byte_offset": self.byte_offset,
            "vocab_size": self.vocab_size,
        }

