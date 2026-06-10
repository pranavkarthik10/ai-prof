import unittest

import numpy as np

from ai_prof.rtc import _has_speech


class VoiceGateTests(unittest.TestCase):
    def test_silence_is_rejected(self):
        self.assertFalse(_has_speech(np.zeros(16_000, dtype=np.float32)))

    def test_audible_signal_is_accepted(self):
        t = np.arange(16_000, dtype=np.float32) / 16_000
        signal = 0.05 * np.sin(2 * np.pi * 220 * t)
        self.assertTrue(_has_speech(signal))

    def test_int16_audio_is_normalized(self):
        signal = np.full(2_000, 1_000, dtype=np.int16)
        self.assertTrue(_has_speech(signal))


if __name__ == "__main__":
    unittest.main()
