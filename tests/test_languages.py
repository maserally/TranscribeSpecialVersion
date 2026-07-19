import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from asr_stage import build_sentences
from studio.main import _output_path, app, open_output
from studio.providers import normalize_openai_base_url
from studio.schemas import JobOptions, ProviderSettings
from studio.subtitles import write_subtitles
from studio.translation import audit_translation, safe_high_risk, translate_cues


class LanguageSupportTests(unittest.TestCase):
    def test_openai_base_url_accepts_root_v1_and_complete_endpoints(self):
        expected = "https://letsgoapi.com/v1"
        self.assertEqual(normalize_openai_base_url("https://letsgoapi.com"), expected)
        self.assertEqual(normalize_openai_base_url("https://letsgoapi.com/v1/"), expected)
        self.assertEqual(
            normalize_openai_base_url(
                "https://letsgoapi.com/v1/audio/transcriptions"
            ),
            expected,
        )
        self.assertEqual(
            normalize_openai_base_url("https://letsgoapi.com/v1/chat/completions"),
            expected,
        )

    def test_job_options_accept_supported_languages(self):
        self.assertEqual(JobOptions(input_path="movie.mp4").source_language, "ja")
        self.assertEqual(
            JobOptions(input_path="movie.mp4", source_language="ko").source_language,
            "ko",
        )
        with self.assertRaises(ValidationError):
            JobOptions(input_path="movie.mp4", source_language="en")

    def test_job_options_clean_quoted_input_paths(self):
        expected = r"E:\Videos\示例影片\movie.mp4"
        self.assertEqual(JobOptions(input_path=f'"{expected}"').input_path, expected)
        self.assertEqual(JobOptions(input_path=f" “{expected}” ").input_path, expected)

    def test_korean_sentence_builder_preserves_spaces(self):
        words = [
            {"word": "안녕하세요", "start": 0.1, "end": 0.5, "probability": 0.9},
            {"word": "여러분", "start": 0.55, "end": 0.9, "probability": 0.9},
        ]
        rows = build_sentences(words, "ko")
        self.assertEqual(rows[0]["source"], "안녕하세요 여러분")

    def test_korean_high_risk_audit(self):
        self.assertTrue(audit_translation("하지 마", "继续", "ko"))
        self.assertEqual(audit_translation("하지 마", "住手", "ko"), [])
        self.assertEqual(safe_high_risk("기다려", "", "ko"), "等一下")
        self.assertEqual(safe_high_risk("살려 줘", "", "ko"), "救救我")

    def test_translation_uses_selected_language_and_accepts_legacy_rows(self):
        class FakeProvider:
            def __init__(self):
                self.prompts = []
                self.requests = []

            def chat_json(self, model, prompt, request):
                self.prompts.append(prompt)
                self.requests.append(request)
                return {"id": request["target"]["id"], "zh": "你好"}

        fake = FakeProvider()
        settings = ProviderSettings(kind="local_ollama", model="test")
        with patch("studio.translation.provider_from_settings", return_value=fake):
            korean = translate_cues(
                [{"start": 0, "end": 1, "source": "안녕하세요"}],
                settings,
                source_language="ko",
            )
        self.assertEqual(korean[0]["source"], "안녕하세요")
        self.assertIn("韩中影视字幕译者", fake.prompts[0])
        self.assertEqual(fake.requests[0]["target"]["source"], "안녕하세요")

        with patch("studio.translation.provider_from_settings", return_value=FakeProvider()):
            japanese = translate_cues(
                [{"start": 0, "end": 1, "ja": "こんにちは"}], settings
            )
        self.assertEqual(japanese[0]["source"], "こんにちは")
        self.assertNotIn("ja", japanese[0])

    def test_korean_output_names_and_content(self):
        cues = [{"start": 0.0, "end": 1.0, "source": "안녕하세요", "zh": "你好"}]
        with tempfile.TemporaryDirectory() as folder:
            files = write_subtitles(cues, Path(folder), "sample", "观看版", "ko")
            self.assertIn("韩文字幕", files["source_srt"].name)
            self.assertIn("中韩双语", files["bilingual_srt"].name)
            text = files["bilingual_srt"].read_text(encoding="utf-8-sig")
            self.assertIn("안녕하세요", text)

    def test_local_open_routes_are_registered(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/jobs/{job_id}/open/{output_key}", paths)
        self.assertIn("/api/jobs/{job_id}/open-folder", paths)

    def test_local_open_rejects_paths_outside_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_root = root / "safe-job"
            output = job_root / "output" / "subtitle.srt"
            output.parent.mkdir(parents=True)
            output.write_text("subtitle", encoding="utf-8")
            job = SimpleNamespace(outputs={"subtitle": str(output)})
            with patch("studio.main.JOBS_DIR", root), patch(
                "studio.main.manager.get", return_value=job
            ):
                self.assertEqual(_output_path("safe-job", "subtitle"), output.resolve())
                with patch("studio.main._open_local") as opener:
                    self.assertTrue(open_output("safe-job", "subtitle")["ok"])
                    opener.assert_called_once_with(output.resolve())

            outside = root / "outside.srt"
            outside.write_text("outside", encoding="utf-8")
            unsafe_job = SimpleNamespace(outputs={"subtitle": str(outside)})
            with patch("studio.main.JOBS_DIR", root), patch(
                "studio.main.manager.get", return_value=unsafe_job
            ), self.assertRaises(HTTPException) as raised:
                _output_path("safe-job", "subtitle")
            self.assertEqual(raised.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
