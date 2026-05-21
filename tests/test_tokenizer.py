import unittest

from helixdiff.tokenizer import ByteTokenizer


class TokenizerTest(unittest.TestCase):
    def test_round_trip_utf8(self) -> None:
        tokenizer = ByteTokenizer()
        text = "diffusion repairs bytes: café\n"
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(tokenizer.vocab_size, 260)


if __name__ == "__main__":
    unittest.main()

