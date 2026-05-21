from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from helixdiff.export import export_checkpoint
from helixdiff.model import DiffusionTransformer, ModelConfig, count_parameters
from helixdiff.sample import load_checkpoint
from helixdiff.tokenizer import ByteTokenizer
from helixdiff.train import TrainConfig, save_checkpoint


class ExportTests(unittest.TestCase):
    def test_export_writes_slim_loadable_checkpoint(self) -> None:
        tokenizer = ByteTokenizer()
        model = DiffusionTransformer(
            ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=16, dim=16, layers=1, heads=2),
            pad_token_id=tokenizer.pad_token_id,
        )
        ema_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "full.pt"
            out = Path(tmp) / "slim.pt"
            save_checkpoint(
                src,
                model,
                TrainConfig(seq_len=16, dim=16, layers=1, heads=2),
                tokenizer,
                step=3,
                metrics={"parameters": count_parameters(model)},
                sample_token_ids=[tokenizer.eos_token_id, tokenizer.byte_offset + ord("a")],
                ema_model_state=ema_state,
            )

            report = export_checkpoint(src, out)
            payload = torch.load(out, map_location="cpu", weights_only=False)
            loaded_model, _, loaded_payload = load_checkpoint(out, device="cpu", use_ema=False)

        self.assertEqual(report["exported_state"], "ema_model_state")
        self.assertIsNone(payload["optimizer_state"])
        self.assertIsNone(payload["ema_model_state"])
        self.assertEqual(loaded_payload["loaded_state"], "model_state")
        self.assertEqual(count_parameters(loaded_model), count_parameters(model))
        self.assertIn("clock_conditioned_denoiser", payload["novel_mechanisms"])


if __name__ == "__main__":
    unittest.main()
