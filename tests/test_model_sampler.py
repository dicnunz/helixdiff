from pathlib import Path
import tempfile
import unittest

import torch

from helixdiff.model import DiffusionTransformer, ModelConfig
from helixdiff.ngram import BigramGuide
from helixdiff.sample import generate_text, load_checkpoint
from helixdiff.tokenizer import ByteTokenizer


class ModelSamplerTest(unittest.TestCase):
    def test_forward_and_sample(self) -> None:
        torch.manual_seed(1)
        tokenizer = ByteTokenizer()
        config = ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=32, dim=32, layers=1, heads=4, ff_mult=2, dropout=0.0)
        model = DiffusionTransformer(config, pad_token_id=tokenizer.pad_token_id)
        tokens = torch.tensor([tokenizer.encode("abc", add_bos=True, add_eos=True) + [tokenizer.mask_token_id] * 27])
        tokens = tokens[:, :32]
        logits = model(tokens, torch.tensor([0.5]))
        self.assertEqual(logits.shape, (1, 32, tokenizer.vocab_size))
        conditioned = model(
            tokens,
            torch.tensor([0.5]),
            corruption_mode=torch.tensor([3]),
            mask_fraction=torch.tensor([0.75]),
        )
        self.assertEqual(conditioned.shape, logits.shape)
        text = generate_text(model, tokenizer, prompt="a", total_tokens=24, steps=4, top_k=16, seed=4)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        ribbon = generate_text(model, tokenizer, prompt="a", total_tokens=24, steps=12, top_k=16, seed=4, schedule="ribbon")
        self.assertIsInstance(ribbon, str)

    def test_bigram_guided_sample(self) -> None:
        torch.manual_seed(2)
        tokenizer = ByteTokenizer()
        config = ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=32, dim=32, layers=1, heads=4, ff_mult=2, dropout=0.0)
        model = DiffusionTransformer(config, pad_token_id=tokenizer.pad_token_id)
        guide = BigramGuide.from_text("abababab", tokenizer)
        text = generate_text(model, tokenizer, prompt="a", total_tokens=24, steps=4, top_k=16, seed=5, guide=guide, guidance=0.5)
        self.assertIsInstance(text, str)
        scaffolded = generate_text(
            model,
            tokenizer,
            prompt="a",
            total_tokens=24,
            steps=4,
            top_k=16,
            seed=5,
            guide=guide,
            guidance=0.5,
            scaffold=True,
            scaffold_remask=0.2,
        )
        self.assertIsInstance(scaffolded, str)

    def test_ngram_guide_uses_high_order_left_anchor(self) -> None:
        tokenizer = ByteTokenizer()
        guide = BigramGuide.from_text("alpha sentence, alpha signal,", tokenizer, order=8)
        prefix = tokenizer.encode("alpha sen", add_bos=True, add_eos=False)
        tokens = torch.tensor([prefix + [tokenizer.mask_token_id]])
        logits = guide.logits(tokens, tokenizer)
        predicted = int(logits[0, -1].argmax().item())
        self.assertEqual(predicted, tokenizer.encode("t", add_bos=False, add_eos=False)[0])

    def test_load_checkpoint_prefers_ema_when_present(self) -> None:
        tokenizer = ByteTokenizer()
        config = ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=16, dim=16, layers=1, heads=4, ff_mult=2, dropout=0.0)
        model = DiffusionTransformer(config, pad_token_id=tokenizer.pad_token_id)
        raw = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
        ema = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ema.pt"
            torch.save(
                {
                    "model_config": config.to_dict(),
                    "model_state": raw,
                    "ema_model_state": ema,
                    "sample_token_ids": list(range(tokenizer.vocab_size)),
                },
                checkpoint,
            )
            loaded, _, payload = load_checkpoint(checkpoint)
            self.assertEqual(payload["loaded_state"], "ema_model_state")
            self.assertAlmostEqual(float(loaded.token_emb.weight.mean().item()), 1.0)
            loaded_raw, _, raw_payload = load_checkpoint(checkpoint, use_ema=False)
            self.assertEqual(raw_payload["loaded_state"], "model_state")
            self.assertAlmostEqual(float(loaded_raw.token_emb.weight.mean().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
