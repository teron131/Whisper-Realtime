import os
import unittest

import numpy as np
import soundfile as sf

from whisper_live.utils import resample
from whisper_live.vad import VoiceActivityDetector


class TestVoiceActivityDetection(unittest.TestCase):
    def setUp(self):
        self.vad = VoiceActivityDetector()
        self.sample_rate = 16000

    def generate_silence(self, duration_seconds):
        return np.zeros(int(self.sample_rate * duration_seconds), dtype=np.float32)

    def load_speech_segment(self, filepath):
        resampled_path = resample(filepath, self.sample_rate)
        audio_data, _ = sf.read(resampled_path, dtype="float32")
        if os.path.exists(resampled_path):
            os.remove(resampled_path)
        return audio_data

    def test_vad_silence_detection(self):
        silence = self.generate_silence(3)
        is_speech_present = self.vad(silence.copy())
        self.assertFalse(is_speech_present, "VAD incorrectly identified silence as speech.")

    def test_vad_speech_detection(self):
        audio_data = self.load_speech_segment("assets/jfk.flac")
        is_speech_present = self.vad(audio_data)
        self.assertTrue(is_speech_present, "VAD failed to identify speech segment.")
