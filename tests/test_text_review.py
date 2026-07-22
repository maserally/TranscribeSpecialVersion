import unittest
from unittest.mock import patch

from studio.schemas import ProviderSettings
from studio.text_review import review_cues


class TextReviewTests(unittest.TestCase):
    def test_review_uses_twenty_four_cue_batches(self):
        class FakeProvider:
            def __init__(self):
                self.calls = 0

            def chat_json(self, model, prompt, request):
                self.calls += 1
                return {
                    "items": [
                        {
                            "id": item["id"],
                            "source": item["source"],
                            "zh": item["zh"],
                            "reason": "保持",
                        }
                        for item in request["targets"]
                    ]
                }

        provider = FakeProvider()
        rows = [
            {"id": index + 1, "start": index, "end": index + 1, "source": "はい", "zh": "好的"}
            for index in range(25)
        ]
        with patch("studio.text_review.provider_from_settings", return_value=provider):
            reviewed, _ = review_cues(
                rows, ProviderSettings(kind="local_ollama", model="review-model")
            )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(reviewed, rows)

    def test_review_preserves_ids_count_and_timing(self):
        class FakeProvider:
            def chat_json(self, model, prompt, request):
                return {
                    "items": [
                        {
                            "id": item["id"],
                            "source": item["source"],
                            "zh": "柿田的照片" if item["id"] == 1 else item["zh"],
                            "reason": "统一人名",
                        }
                        for item in request["targets"]
                    ],
                    "glossary": [{"source": "かきた", "zh": "柿田"}],
                }

        rows = [
            {"id": 1, "start": 1.25, "end": 2.5, "source": "かきたの写真", "zh": "柿田照片"},
            {"id": 2, "start": 3.0, "end": 4.75, "source": "そうです", "zh": "是的"},
        ]
        settings = ProviderSettings(kind="local_ollama", model="review-model")
        with patch("studio.text_review.provider_from_settings", return_value=FakeProvider()):
            reviewed, audit = review_cues(rows, settings)

        self.assertEqual(len(reviewed), len(rows))
        self.assertEqual([row["id"] for row in reviewed], [1, 2])
        self.assertEqual(
            [(row["start"], row["end"]) for row in reviewed],
            [(row["start"], row["end"]) for row in rows],
        )
        self.assertEqual(reviewed[0]["zh"], "柿田的照片")
        self.assertEqual(audit["changed_count"], 1)

    def test_invalid_batch_is_retried_then_keeps_originals(self):
        class InvalidProvider:
            def __init__(self):
                self.calls = 0

            def chat_json(self, model, prompt, request):
                self.calls += 1
                return {"items": [{"id": 999, "source": "错误", "zh": "错误"}]}

        provider = InvalidProvider()
        rows = [{"id": 1, "start": 0, "end": 1, "source": "はい", "zh": "好的"}]
        settings = ProviderSettings(kind="local_ollama", model="review-model")
        with patch("studio.text_review.provider_from_settings", return_value=provider):
            reviewed, audit = review_cues(rows, settings)

        self.assertEqual(provider.calls, 2)
        self.assertEqual(reviewed, rows)
        self.assertEqual(audit["invalid_batches"], 1)

    def test_review_that_introduces_semantic_risk_is_reverted(self):
        class RiskyProvider:
            def chat_json(self, model, prompt, request):
                return {
                    "items": [
                        {"id": 1, "source": "待って", "zh": "继续", "reason": "错误改写"}
                    ]
                }

        rows = [{"id": 1, "start": 0, "end": 1, "source": "待って", "zh": "等一下"}]
        settings = ProviderSettings(kind="local_ollama", model="review-model")
        with patch("studio.text_review.provider_from_settings", return_value=RiskyProvider()):
            reviewed, audit = review_cues(rows, settings)

        self.assertEqual(reviewed[0]["zh"], "等一下")
        self.assertEqual(audit["changed_count"], 0)
        self.assertEqual(audit["rejected_count"], 1)


if __name__ == "__main__":
    unittest.main()
