import unittest
from unittest.mock import patch

from ai_prof.config import ModelConfig
from ai_prof.warmup import _service_url, start_service_warmup


class WarmupTests(unittest.TestCase):
    def test_service_url_removes_v1_suffix(self):
        config = ModelConfig("https://example.modal.run/v1", "key", "model")
        self.assertEqual(
            _service_url(config, "/health"),
            "https://example.modal.run/health",
        )

    def test_service_url_skips_unconfigured_service(self):
        config = ModelConfig(None, "key", "model")
        self.assertIsNone(_service_url(config, "/health"))

    @patch("ai_prof.warmup.threading.Thread")
    def test_warmup_starts_only_once(self, thread):
        import ai_prof.warmup as warmup

        warmup._started = False
        start_service_warmup()
        start_service_warmup()

        thread.assert_called_once()
        thread.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
