import unittest

import torch

from helixdiff.infill import parse_marked_infill
from helixdiff.model import DiffusionTransformer, ModelConfig
from helixdiff.sample import denoise_ids, render_tokens_with_masks
from helixdiff.tokenizer import ByteTokenizer


class InfillTest(unittest.TestCase):
    def test_parse_marked_infill_masks_exact_hole_bytes(self) -> None:
        tokenizer = ByteTokenizer()
        example = parse_marked_infill("repair [[this]] span", tokenizer)
        self.assertEqual(example.hole, "this")
        self.assertEqual(example.hole_length, 4)
        self.assertEqual(int((example.tokens == tokenizer.mask_token_id).sum().item()), 4)
        self.assertIn("~~~~", render_tokens_with_masks(example.tokens, tokenizer))

    def test_denoise_ids_preserves_frozen_context_and_emits_trace(self) -> None:
        torch.manual_seed(3)
        tokenizer = ByteTokenizer()
        config = ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=40, dim=32, layers=1, heads=4, ff_mult=2, dropout=0.0)
        model = DiffusionTransformer(config, pad_token_id=tokenizer.pad_token_id)
        example = parse_marked_infill("alpha [[beta]] gamma", tokenizer)
        repaired, trace = denoise_ids(
            model,
            tokenizer,
            initial_tokens=example.tokens,
            frozen=example.frozen,
            steps=5,
            top_k=16,
            seed=11,
            return_trace=True,
        )
        self.assertIsInstance(repaired, torch.Tensor)
        self.assertTrue((repaired[example.frozen] == example.target[example.frozen]).all().item())
        self.assertFalse((repaired == tokenizer.mask_token_id).any().item())
        self.assertGreaterEqual(len(trace), 1)
        self.assertEqual(trace[-1]["remaining_after"], 0)
        self.assertIn("mean_masked_entropy", trace[0])


if __name__ == "__main__":
    unittest.main()
