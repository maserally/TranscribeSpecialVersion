import unittest

from audio_event_gate import vad_speech_fallback


class AudioEventGateTests(unittest.TestCase):
    def test_vad_fallback_keeps_every_candidate_for_whisper(self):
        units = [{"start": 1.0, "end": 2.5}, {"start": 4.0, "end": 6.0}]

        rows = vad_speech_fallback(units, "offline test")

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["speech_score"] == 1.0 for row in rows))
        self.assertTrue(all(row["event_gate_fallback"] for row in rows))
        self.assertEqual(rows[0]["start"], 1.0)


if __name__ == "__main__":
    unittest.main()
