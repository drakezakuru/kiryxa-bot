"""Tests for VisionClient (image analysis, reaction selection, comment generation)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.vision_client import ALLOWED_REACTIONS, VisionClient


class TestVisionClient(unittest.TestCase):
    def test_allowed_reactions(self) -> None:
        self.assertIn("🔥", ALLOWED_REACTIONS)
        self.assertIn("🗿", ALLOWED_REACTIONS)
        self.assertIn("🤨", ALLOWED_REACTIONS)
        self.assertIn("😂", ALLOWED_REACTIONS)

    def test_analyze_image_mock(self) -> None:
        import core.vision_client
        import json

        vc = VisionClient(api_key="test_key")

        # Mock urllib to return valid JSON
        import urllib.request
        orig_urlopen = urllib.request.urlopen

        class MockResp:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps({
                    "choices": [{
                        "message": {
                            "content": json.dumps({"reaction": "🔥", "comment": "фурр~ зачетный артик!"})
                        }
                    }]
                }).encode("utf-8")

        try:
            urllib.request.urlopen = lambda *args, **kwargs: MockResp()
            res = vc.analyze_image(b"fake_bytes")
            self.assertIsNotNone(res)
            self.assertEqual(res[0], "🔥")
            self.assertIn("артик", res[1])
        finally:
            urllib.request.urlopen = orig_urlopen


if __name__ == "__main__":
    unittest.main()
