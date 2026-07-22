import unittest
from unittest.mock import patch

from studio.quality import finalize_cues, quality_summary
from studio.asr_context import attach_asr_reviews
from studio.recall import (
    accepted_recovery_rows,
    filter_events_for_uncovered_speech,
    vad_fallback_events_for_gaps,
)
from studio.schemas import ProviderSettings
from studio.translation import (
    build_translation_plan,
    audit_translation,
    safe_high_risk,
    translate_cues,
)


class QualityOptimizationTests(unittest.TestCase):
    def test_dense_short_cues_are_merged_without_overlap(self):
        rows = finalize_cues(
            [
                {"start": 0.0, "end": 0.3, "source": "あ", "zh": "啊"},
                {"start": 0.1, "end": 0.95, "source": "や", "zh": "呀"},
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["zh"], "啊，呀")
        self.assertGreaterEqual(rows[0]["end"], 0.9)

    def test_adjacent_exact_duplicates_are_collapsed(self):
        rows = finalize_cues(
            [
                {"start": 0.0, "end": 0.5, "source": "はい", "zh": "好的"},
                {"start": 0.54, "end": 1.2, "source": "はい", "zh": "好的"},
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["zh"], "好的")

    def test_same_start_cues_cannot_leave_an_overlap(self):
        rows = finalize_cues(
            [
                {"start": 2.0, "end": 2.5, "source": "第一句", "zh": "这是一条很长很长的第一句字幕内容"},
                {"start": 2.0, "end": 3.0, "source": "第二句", "zh": "这是一条很长很长的第二句字幕内容"},
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(sum(rows[i]["end"] > rows[i + 1]["start"] for i in range(len(rows) - 1)), 0)

    def test_review_marks_translation_warnings_but_publish_does_not(self):
        cue = {
            "start": 0.0,
            "end": 1.0,
            "source": "かきたちゃん",
            "zh": "柿田酱",
            "translation_warnings": ["人名需复核"],
        }
        review = finalize_cues([cue], publish=False)
        publish = finalize_cues([cue], publish=True)
        self.assertTrue(review[0]["zh"].startswith("【需校对】"))
        self.assertNotIn("【需校对】", publish[0]["zh"])

    def test_quality_summary_measures_vad_activity_coverage(self):
        summary = quality_summary(
            [{"start": 1.0, "end": 3.0, "zh": "对白"}],
            40.0,
            activity_segments=[
                {"start": 0.0, "end": 2.0},
                {"start": 4.0, "end": 6.0},
            ],
        )
        self.assertEqual(summary["activity_seconds"], 4.0)
        self.assertEqual(summary["covered_activity_seconds"], 1.0)
        self.assertEqual(summary["activity_coverage_percent"], 25.0)

    def test_vad_fallback_only_returns_unreviewed_gap_pieces(self):
        rows = vad_fallback_events_for_gaps(
            [{"start": 10.0, "end": 14.0}, {"start": 30.0, "end": 32.0}],
            [{"start": 9.0, "end": 20.0, "duration": 11.0}],
            [{"start": 11.0, "end": 13.0}],
        )
        self.assertEqual([(x["start"], x["end"]) for x in rows], [(10.0, 11.0), (13.0, 14.0)])
        self.assertTrue(all(x["source"] == "vad_gap_fallback" for x in rows))

    def test_recovery_provenance_does_not_replace_recognized_text(self):
        rows = accepted_recovery_rows(
            [{"start": 1.0, "end": 2.0, "source": "待って"}],
            [{"start": 1.0, "end": 2.0, "similarity": 0.9}],
            0.5,
        )
        self.assertEqual(rows[0]["source"], "待って")
        self.assertEqual(rows[0]["recovery_source"], "gap_recovery_consensus")
        rejected = accepted_recovery_rows(
            [
                {
                    "start": 1.0,
                    "end": 2.0,
                    "source": "待って",
                    "review_source": "medium_fallback",
                }
            ],
            [{"start": 1.0, "end": 2.0, "similarity": 0.9}],
            0.5,
        )
        self.assertEqual(rejected, [])

    def test_music_recovery_finds_short_uncovered_speech_only(self):
        events = [
            {"start": 1.0, "end": 2.0, "speech_score": 0.12, "nonlexical_score": 0.04},
            {"start": 4.0, "end": 5.0, "speech_score": 0.11, "nonlexical_score": 0.04},
            {"start": 7.0, "end": 8.0, "speech_score": 0.03, "nonlexical_score": 0.02},
        ]
        rows = filter_events_for_uncovered_speech(
            events, [{"start": 0.8, "end": 2.2}], 0.08, 1.2
        )
        self.assertEqual([(x["start"], x["end"]) for x in rows], [(4.0, 5.0)])
        self.assertEqual(rows[0]["source"], "music_recovery_candidate")

    def test_asr_disagreement_is_attached_for_translation_context(self):
        rows = attach_asr_reviews(
            [{"start": 10.0, "end": 12.0, "source": "一本でもごぼう"}],
            [{"start": 10.0, "end": 12.0, "large_v3": "一本でもごぼう", "medium": "一本でもニンジン", "similarity": 0.5}],
        )
        self.assertTrue(rows[0]["asr_review"]["disagreement"])
        self.assertEqual(rows[0]["asr_review"]["alternative"], "一本でもニンジン")

    def test_naturalness_audit_catches_reported_bad_literal_translations(self):
        self.assertTrue(audit_translation("毎日拝むぜ", "我每天都要来朝拜啊", "ja"))
        self.assertTrue(
            audit_translation("一本でもごぼう、ごぼうでも人参", "即使是一根牛蒡，牛蒡也是胡萝卜", "ja")
        )

    def test_lexical_nai_adjectives_are_not_mistaken_for_negation(self):
        self.assertEqual(audit_translation("危ない", "危险", "ja"), [])
        self.assertEqual(audit_translation("もったいない", "太可惜了", "ja"), [])

    def test_japanese_action_direction_is_audited_and_has_safe_fallback(self):
        self.assertTrue(audit_translation("あ、抜けちゃった", "啊，搞砸了", "ja"))
        self.assertEqual(safe_high_risk("あ、抜けちゃった", "啊，搞砸了"), "啊，掉出来了")
        self.assertTrue(audit_translation("入れます", "不要这样", "ja"))
        self.assertEqual(safe_high_risk("入れます", "不要这样"), "要放进去了")

    def test_source_script_gets_a_focused_third_repair(self):
        class FakeProvider:
            def __init__(self):
                self.calls = 0

            def chat_json(self, model, prompt, request):
                self.calls += 1
                ids = [target["id"] for target in request["targets"]]
                if self.calls < 3:
                    return {"items": [{"id": item_id, "zh": "我在看かきた酱的照片"} for item_id in ids]}
                return {"items": [{"id": item_id, "zh": "我在看柿田酱的照片"} for item_id in ids]}

        provider = FakeProvider()
        settings = ProviderSettings(kind="local_ollama", model="test")
        with patch("studio.translation.provider_from_settings", return_value=provider):
            rows = translate_cues(
                [{"start": 0, "end": 1, "source": "かきたちゃんの写真を見てる"}],
                settings,
            )
        self.assertEqual(provider.calls, 3)
        self.assertEqual(rows[0]["zh"], "我在看柿田酱的照片")
        self.assertNotIn("translation_warnings", rows[0])

    def test_context_translation_batches_twelve_cues_per_request(self):
        class FakeProvider:
            def __init__(self):
                self.calls = 0

            def chat_json(self, model, prompt, request):
                self.calls += 1
                return {
                    "items": [
                        {"id": target["id"], "zh": "你好"}
                        for target in request["targets"]
                    ]
                }

        provider = FakeProvider()
        rows = [
            {"start": index, "end": index + 1, "source": "こんにちは"}
            for index in range(25)
        ]
        with patch("studio.translation.provider_from_settings", return_value=provider):
            translated = translate_cues(
                rows, ProviderSettings(kind="local_ollama", model="test")
            )
        self.assertEqual(provider.calls, 3)
        self.assertEqual(len(translated), 25)

    def test_invalid_large_translation_batch_splits_without_losing_cues(self):
        class SplitProvider:
            def chat_json(self, model, prompt, request):
                if len(request["targets"]) > 6:
                    return {"items": []}
                return {
                    "items": [
                        {"id": target["id"], "zh": "好的"}
                        for target in request["targets"]
                    ]
                }

        rows = [
            {"start": index, "end": index + 1, "source": "はい"}
            for index in range(12)
        ]
        with patch(
            "studio.translation.provider_from_settings", return_value=SplitProvider()
        ):
            translated = translate_cues(
                rows, ProviderSettings(kind="local_ollama", model="test")
            )
        self.assertEqual(len(translated), 12)
        self.assertTrue(all(row["zh"] == "好的" for row in translated))

    def test_translation_plan_summarizes_large_source_in_scene_chunks(self):
        class FakeProvider:
            def __init__(self):
                self.calls = 0

            def chat_json(self, model, prompt, request):
                self.calls += 1
                return {
                    "summary": f"场景{self.calls}",
                    "characters": ["人物甲"],
                    "glossary": [{"source": "甲", "zh": "甲"}],
                }

        provider = FakeProvider()
        rows = [{"source": "はい"} for _ in range(241)]
        with patch("studio.translation.provider_from_settings", return_value=provider):
            plan = build_translation_plan(
                rows, ProviderSettings(kind="local_ollama", model="test")
            )
        self.assertEqual(provider.calls, 3)
        self.assertEqual(plan["cue_count"], 241)
        self.assertEqual(len(plan["scenes"]), 3)


if __name__ == "__main__":
    unittest.main()
