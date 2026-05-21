import tempfile
import unittest
from pathlib import Path

from helixdiff.verify_scratch import scan_repo


class VerifyScratchTest(unittest.TestCase):
    def test_scanner_finds_pretrained_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = "model = " + "Auto" + "Model" + "." + "from" + "_pretrained" + "('x')\n"
            (root / "bad.py").write_text(bad, encoding="utf-8")
            findings = scan_repo(root)
            self.assertTrue(findings)


if __name__ == "__main__":
    unittest.main()
