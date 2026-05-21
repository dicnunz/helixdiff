import unittest

import torch

from helixdiff.adapt import (
    VisibleAdaptConfig,
    adapt_model_to_visible_context,
    make_visible_suture_batch,
    visible_context_text,
)
from helixdiff.infill import parse_marked_infill
from helixdiff.model import DiffusionTransformer, ModelConfig
from helixdiff.tokenizer import ByteTokenizer


class VisibleAdaptTest(unittest.TestCase):
    def test_visible_context_omits_marked_hole(self) -> None:
        tokenizer = ByteTokenizer()
        example = parse_marked_infill("alpha [[secret]] omega", tokenizer)
        self.assertEqual(visible_context_text(example), "alpha  omega")
        self.assertNotIn("secret", visible_context_text(example))

    def test_visible_suture_batch_masks_only_repairable_bytes(self) -> None:
        tokenizer = ByteTokenizer()
        generator = torch.Generator(device="cpu").manual_seed(3)
        clean, corrupted, mask = make_visible_suture_batch(
            visible_text="abcdef ghijkl mnopqr",
            tokenizer=tokenizer,
            seq_len=32,
            batch_size=2,
            span_min=2,
            span_max=4,
            device=torch.device("cpu"),
            generator=generator,
        )
        self.assertEqual(clean.shape, corrupted.shape)
        self.assertEqual(mask.shape, clean.shape)
        self.assertTrue(mask.any().item())
        self.assertTrue(torch.all(corrupted[mask] == tokenizer.mask_token_id).item())
        self.assertFalse(torch.any(clean[mask] == tokenizer.bos_token_id).item())
        self.assertFalse(torch.any(clean[mask] == tokenizer.eos_token_id).item())

    def test_adaptation_returns_session_copy_and_leak_flag(self) -> None:
        tokenizer = ByteTokenizer()
        model = DiffusionTransformer(
            ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=48, dim=24, layers=1, heads=2, dropout=0.0),
            pad_token_id=tokenizer.pad_token_id,
        )
        model.allowed_token_ids = list(range(tokenizer.vocab_size))
        example = parse_marked_infill("alpha beta [[secret]] gamma delta alpha beta", tokenizer)
        adapted, report = adapt_model_to_visible_context(
            model=model,
            tokenizer=tokenizer,
            example=example,
            config=VisibleAdaptConfig(steps=1, batch_size=2, learning_rate=1e-4, train_scope="head", seed=4),
        )
        self.assertIsNot(adapted, model)
        self.assertTrue(report["enabled"])
        self.assertFalse(report["hidden_target_seen_in_visible_context"])
        self.assertGreater(report["trainable_parameters"], 0)
        self.assertIn("visible_context_sha256", report)


if __name__ == "__main__":
    unittest.main()
