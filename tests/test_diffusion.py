import unittest

import torch

from helixdiff.diffusion import corrupt_batch, masked_cross_entropy, restrict_logits_to_ids, suture_boundary_weights
from helixdiff.tokenizer import ByteTokenizer


class DiffusionTest(unittest.TestCase):
    def test_corruption_masks_valid_tokens(self) -> None:
        tokenizer = ByteTokenizer()
        tokens = torch.tensor([tokenizer.encode("hello world", add_bos=True, add_eos=True)])
        corrupted, mask, rates = corrupt_batch(tokens, tokenizer, t=torch.tensor([0.7]))
        self.assertEqual(corrupted.shape, tokens.shape)
        self.assertTrue(mask.any().item())
        self.assertFalse(mask[0, 0].item())
        self.assertFalse(mask[0, -1].item())
        self.assertGreater(float(rates.item()), 0.0)

    def test_masked_loss(self) -> None:
        tokenizer = ByteTokenizer()
        targets = torch.tensor([[tokenizer.bos_token_id, tokenizer.byte_offset + 1]])
        logits = torch.randn(1, 2, tokenizer.vocab_size)
        mask = torch.tensor([[False, True]])
        loss = masked_cross_entropy(logits, targets, mask)
        self.assertTrue(torch.isfinite(loss).item())

    def test_masked_loss_accepts_weights(self) -> None:
        tokenizer = ByteTokenizer()
        targets = torch.tensor([[tokenizer.byte_offset + 1, tokenizer.byte_offset + 2]])
        logits = torch.randn(1, 2, tokenizer.vocab_size)
        mask = torch.tensor([[True, True]])
        weights = torch.tensor([[2.0, 1.0]])
        loss = masked_cross_entropy(logits, targets, mask, weights=weights)
        self.assertTrue(torch.isfinite(loss).item())

    def test_restrict_logits_to_active_ids(self) -> None:
        logits = torch.zeros(1, 1, 8)
        restricted = restrict_logits_to_ids(logits, [2, 5])
        self.assertTrue(torch.isfinite(restricted[..., 2]).item())
        self.assertLess(float(restricted[..., 3].item()), -1e30)

    def test_ribbon_corruption_masks_suffix(self) -> None:
        tokenizer = ByteTokenizer()
        tokens = torch.tensor([tokenizer.encode("abcdefgh", add_bos=True, add_eos=True)])
        _, mask, _ = corrupt_batch(
            tokens,
            tokenizer,
            t=torch.tensor([0.0]),
            min_mask_rate=0.0,
            max_mask_rate=0.0,
            span_prob=0.0,
            ribbon_prob=1.0,
        )
        masked_positions = torch.nonzero(mask[0], as_tuple=False).flatten().tolist()
        self.assertGreater(len(masked_positions), 0)
        self.assertEqual(masked_positions, list(range(masked_positions[0], len(tokens[0]) - 1)))

    def test_suture_corruption_masks_one_bounded_span(self) -> None:
        tokenizer = ByteTokenizer()
        tokens = torch.tensor([tokenizer.encode("abcdefghijklmnop", add_bos=True, add_eos=True)])
        _, mask, _ = corrupt_batch(
            tokens,
            tokenizer,
            t=torch.tensor([0.0]),
            min_mask_rate=0.0,
            max_mask_rate=0.0,
            span_prob=0.0,
            ribbon_prob=0.0,
            suture_prob=1.0,
            suture_min_span=4,
            suture_max_span=4,
        )
        masked_positions = torch.nonzero(mask[0], as_tuple=False).flatten().tolist()
        self.assertEqual(len(masked_positions), 4)
        self.assertEqual(masked_positions, list(range(masked_positions[0], masked_positions[0] + 4)))
        self.assertGreater(masked_positions[0], 0)
        self.assertLess(masked_positions[-1], len(tokens[0]) - 1)

    def test_suture_boundary_weights_only_boosts_suture_edges(self) -> None:
        mask = torch.tensor([[False, True, True, True, False], [False, True, True, False, False]])
        modes = torch.tensor([3, 1])
        weights = suture_boundary_weights(mask, modes, boundary_weight=2.5)
        self.assertEqual(weights[0].tolist(), [1.0, 2.5, 1.0, 2.5, 1.0])
        self.assertEqual(weights[1].tolist(), [1.0, 1.0, 1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
