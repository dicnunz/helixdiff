import unittest

import torch

from helixdiff.infill import score_repair
from helixdiff.tokenizer import ByteTokenizer


class _ProbeModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size
        self.mask_counts: list[list[int]] = []

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        *,
        corruption_mode: torch.Tensor | int | None = None,
        mask_fraction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.mask_counts.append((tokens == ByteTokenizer().mask_token_id).sum(dim=1).tolist())
        logits = torch.zeros(tokens.shape[0], tokens.shape[1], self.vocab_size, device=tokens.device)
        logits[..., ByteTokenizer().byte_offset + ord("a")] = 4.0
        return logits


class InfillScoringTest(unittest.TestCase):
    def test_suture_score_uses_leave_one_out_hole_probes(self) -> None:
        tokenizer = ByteTokenizer()
        repaired = torch.tensor(tokenizer.encode("xxaaay", add_bos=True, add_eos=True), dtype=torch.long)
        start = 3
        end = 6
        model = _ProbeModel(tokenizer.vocab_size)
        score = score_repair(
            model=model,
            tokenizer=tokenizer,
            repaired_ids=repaired,
            hole_start=start,
            hole_end=end,
            guide=None,
            guidance=0.0,
            temperature=1.0,
            top_k=16,
            mode="suture_loo",
        )
        self.assertTrue(torch.isfinite(torch.tensor(score)).item())
        self.assertEqual(model.mask_counts[-1], [1, 1, 1])

    def test_full_hole_score_keeps_legacy_blank_hole_probe(self) -> None:
        tokenizer = ByteTokenizer()
        repaired = torch.tensor(tokenizer.encode("xxaaay", add_bos=True, add_eos=True), dtype=torch.long)
        model = _ProbeModel(tokenizer.vocab_size)
        score_repair(
            model=model,
            tokenizer=tokenizer,
            repaired_ids=repaired,
            hole_start=3,
            hole_end=6,
            guide=None,
            guidance=0.0,
            temperature=1.0,
            top_k=16,
            mode="full_hole",
        )
        self.assertEqual(model.mask_counts[-1], [3])


if __name__ == "__main__":
    unittest.main()
