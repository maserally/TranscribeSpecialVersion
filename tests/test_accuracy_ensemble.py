import unittest

from ensemble_common import (
    choose_consensus,
    needs_third_vote,
    normalize_transcript,
    review_reasons,
    select_windows,
    transcript_similarity,
)


class AccuracyEnsembleTests(unittest.TestCase):
    def test_normalization_ignores_japanese_punctuation(self):
        self.assertEqual(normalize_transcript("そう、ですね。"), "そうですね")

    def test_agreement_does_not_start_whisper(self):
        self.assertFalse(needs_third_vote("今日は大丈夫です", "今日は大丈夫です。"))

    def test_disagreement_starts_whisper(self):
        self.assertTrue(needs_third_vote("毎日お参りします", "ごぼうは人参です"))

    def test_third_vote_selects_pairwise_consensus(self):
        text, winner, scores = choose_consensus(
            "毎日お参りします", "毎日お祈りします", "毎日お参りします。"
        )
        self.assertEqual(winner, "qwen")
        self.assertEqual(text, "毎日お参りします")
        self.assertGreater(scores["qwen_whisper"], scores["cohere_whisper"])

    def test_low_confidence_reasons_are_automatic(self):
        reasons = review_reasons(
            {"start": 0, "end": 4, "speech_score": 0.3, "qwen_source": "あ"}
        )
        self.assertIn("weak_speech", reasons)
        self.assertIn("too_short_for_window", reasons)

    def test_music_masked_window_is_reviewed(self):
        reasons = review_reasons(
            {
                "start": 0,
                "end": 2,
                "speech_score": 0.7,
                "nonlexical_score": 0.3,
                "qwen_source": "大丈夫です",
            }
        )
        self.assertIn("music_or_noise_masking", reasons)

    def test_windows_merge_without_exceeding_limit(self):
        rows = [
            {"start": 1.0, "end": 2.0, "speech_score": 0.8, "nonlexical_score": 0.1},
            {"start": 2.1, "end": 3.0, "speech_score": 0.7, "nonlexical_score": 0.1},
        ]
        windows = select_windows(rows, 0.1, 1.0)
        self.assertEqual(len(windows), 1)
        self.assertAlmostEqual(windows[0]["start"], 0.78)

    def test_similarity_is_symmetric(self):
        left = transcript_similarity("テストです", "テストでした")
        right = transcript_similarity("テストでした", "テストです")
        self.assertAlmostEqual(left, right)


if __name__ == "__main__":
    unittest.main()
