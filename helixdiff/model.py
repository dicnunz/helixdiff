from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 260
    seq_len: int = 256
    dim: int = 384
    layers: int = 8
    heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    noise_buckets: int = 64
    condition_modes: int = 4

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half, device=t.device, dtype=t.dtype) * -1
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.net(emb)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float) -> None:
        super().__init__()
        hidden = dim * mult
        self.up = nn.Linear(dim, hidden * 2)
        self.down = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * value))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.ff_norm = RMSNorm(config.dim)
        self.attn = SelfAttention(config.dim, config.heads, config.dropout)
        self.ff = FeedForward(config.dim, config.ff_mult, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), key_padding_mask=key_padding_mask))
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, config: ModelConfig, *, pad_token_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.pad_token_id = int(pad_token_id)
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.seq_len, config.dim)
        self.time_emb = TimeEmbedding(config.dim)
        self.noise_emb = nn.Embedding(config.noise_buckets, config.dim)
        self.mode_emb = nn.Embedding(config.condition_modes, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        *,
        corruption_mode: torch.Tensor | int | None = None,
        mask_fraction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, seq]")
        batch, seq_len = tokens.shape
        if seq_len > self.config.seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds config.seq_len {self.config.seq_len}")
        if t.ndim == 0:
            t = t.expand(batch)
        if t.shape[0] != batch:
            raise ValueError("t must have one value per batch row")
        positions = torch.arange(seq_len, device=tokens.device)
        clock = torch.clamp(t if mask_fraction is None else mask_fraction, 0.0, 1.0)
        noise_bucket = torch.round(clock * float(self.config.noise_buckets - 1)).long()
        if corruption_mode is None:
            mode_ids = torch.zeros(batch, dtype=torch.long, device=tokens.device)
        elif isinstance(corruption_mode, int):
            mode_ids = torch.full((batch,), int(corruption_mode), dtype=torch.long, device=tokens.device)
        else:
            mode_ids = corruption_mode.to(device=tokens.device, dtype=torch.long)
            if mode_ids.ndim == 0:
                mode_ids = mode_ids.expand(batch)
        mode_ids = torch.clamp(mode_ids, 0, self.config.condition_modes - 1)
        x = (
            self.token_emb(tokens)
            + self.pos_emb(positions)[None, :, :]
            + self.time_emb(t)[:, None, :]
            + self.noise_emb(noise_bucket)[:, None, :]
            + self.mode_emb(mode_ids)[:, None, :]
        )
        key_padding_mask = tokens == self.pad_token_id
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.lm_head(self.norm(x))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
