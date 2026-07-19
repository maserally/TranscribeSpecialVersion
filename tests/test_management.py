import tempfile
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.main import app
from studio.providers import supports_segment_timestamps
from studio.remote_asr import _remote_packs
from studio.runner import JobControl, JobManager, JobState
from studio.schemas import JobOptions, ProviderSettings
from studio.settings_store import (
    load_provider_settings,
    resolve_provider_api_keys,
    save_provider_settings,
)


class TaskManagementTests(unittest.TestCase):
    def test_pause_and_resume_a_real_child_process(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("studio.runner.JOBS_DIR", root / "jobs"), patch(
                "studio.runner.UPLOADS_DIR", root / "uploads"
            ):
                manager = JobManager()
                job = JobState(
                    id="process-job",
                    options=JobOptions(input_path="movie.mp4"),
                    status="running",
                    stage="测试子进程",
                )
                manager.jobs[job.id] = job
                manager.controls[job.id] = JobControl()
                manager.persist(job)
                errors = []

                def run():
                    try:
                        manager.run_command(
                            job,
                            [
                                sys.executable,
                                "-c",
                                "import time; print('start', flush=True); time.sleep(1.2); print('done')",
                            ],
                        )
                    except Exception as exc:
                        errors.append(exc)

                worker = threading.Thread(target=run)
                worker.start()
                deadline = time.time() + 2
                while manager.controls[job.id].process is None and time.time() < deadline:
                    time.sleep(0.02)
                self.assertIsNotNone(manager.controls[job.id].process)
                manager.pause(job.id)
                time.sleep(0.25)
                self.assertTrue(worker.is_alive())
                manager.resume(job.id)
                worker.join(3)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])

    def test_pause_resume_cancel_and_safe_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = root / "jobs"
            uploads = root / "uploads"
            with patch("studio.runner.JOBS_DIR", jobs), patch(
                "studio.runner.UPLOADS_DIR", uploads
            ):
                manager = JobManager()
                job = JobState(
                    id="safe-job",
                    options=JobOptions(input_path="movie.mp4"),
                    status="running",
                    stage="语音识别",
                )
                manager.jobs[job.id] = job
                manager.controls[job.id] = JobControl()
                manager.persist(job)

                manager.pause(job.id)
                self.assertEqual(job.status, "paused")
                self.assertIn("语音识别", job.stage)
                manager.resume(job.id)
                self.assertEqual(job.status, "running")
                self.assertEqual(job.stage, "语音识别")
                manager.cancel(job.id)
                self.assertEqual(job.status, "canceled")
                with self.assertRaises(RuntimeError):
                    manager.delete(job.id)
                manager.controls[job.id].finished_event.set()
                deleted = manager.delete(job.id)
                self.assertFalse(deleted.exists())
                self.assertIsNone(manager.get(job.id))

    def test_task_control_routes_are_registered(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/jobs/{job_id}/pause", paths)
        self.assertIn("/api/jobs/{job_id}/resume", paths)
        self.assertIn("/api/jobs/{job_id}/cancel", paths)

    def test_task_status_hides_all_provider_api_keys(self):
        options = JobOptions(
            input_path="movie.mp4",
            asr=ProviderSettings(kind="openai_compatible", api_key="asr-secret", model="asr"),
            translator=ProviderSettings(
                kind="openai_compatible", api_key="translator-secret", model="translator"
            ),
            text_reviewer=ProviderSettings(
                kind="openai_compatible", api_key="review-secret", model="reviewer"
            ),
        )
        public = JobState(id="secret-test", options=options).public()
        self.assertEqual(public["options"]["asr"]["api_key"], "")
        self.assertEqual(public["options"]["translator"]["api_key"], "")
        self.assertEqual(public["options"]["text_reviewer"]["api_key"], "")

    def test_provider_settings_are_local_and_api_keys_are_not_plaintext(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "provider_settings.json"
            with patch("studio.settings_store.SETTINGS_PATH", path):
                save_provider_settings(
                    {
                        "asr": {
                            "kind": "openai_compatible",
                            "base_url": "https://example.com/v1",
                            "api_key": "secret-asr-key",
                            "model": "whisper-1",
                        },
                        "translator": {
                            "kind": "openai_compatible",
                            "base_url": "https://example.com/v1",
                            "api_key": "secret-translator-key",
                            "model": "translator-model",
                        },
                        "text_reviewer": {
                            "kind": "openai_compatible",
                            "base_url": "https://review.example.com/v1",
                            "api_key": "secret-review-key",
                            "model": "review-model",
                        },
                        "verifier_model": "large-v3",
                    }
                )
                raw = path.read_text(encoding="utf-8")
                self.assertNotIn("secret-asr-key", raw)
                self.assertNotIn("secret-translator-key", raw)
                self.assertNotIn("secret-review-key", raw)
                loaded = load_provider_settings()
                self.assertEqual(loaded["asr"]["api_key"], "secret-asr-key")
                self.assertEqual(loaded["translator"]["model"], "translator-model")
                self.assertEqual(loaded["text_reviewer"]["api_key"], "secret-review-key")

    def test_cloud_settings_never_persist_keys_and_resolve_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "provider_settings.json"
            with patch("studio.settings_store.SETTINGS_PATH", path), patch(
                "studio.settings_store.SECURE_LOCAL_SECRETS", False
            ), patch.dict(
                "os.environ",
                {
                    "SUBTITLE_ASR_API_KEY": "cloud-asr",
                    "SUBTITLE_TRANSLATOR_API_KEY": "cloud-translator",
                    "SUBTITLE_REVIEWER_API_KEY": "cloud-reviewer",
                },
                clear=False,
            ):
                settings = {
                    "asr": {"kind": "openai_compatible", "base_url": "https://a/v1", "api_key": "do-not-save", "model": "asr"},
                    "translator": {"kind": "openai_compatible", "base_url": "https://t/v1", "api_key": "do-not-save", "model": "translator"},
                    "text_reviewer": {"kind": "openai_compatible", "base_url": "https://r/v1", "api_key": "do-not-save", "model": "reviewer"},
                    "verifier_model": "large-v3",
                }
                save_provider_settings(settings)
                self.assertNotIn("do-not-save", path.read_text(encoding="utf-8"))
                loaded = load_provider_settings(expose_secrets=False)
                self.assertTrue(loaded["translator"]["api_key_configured"])
                self.assertEqual(loaded["translator"]["api_key"], "")
                options = JobOptions(
                    input_path="movie.mp4",
                    asr=ProviderSettings(kind="openai_compatible", model="asr"),
                    translator=ProviderSettings(kind="openai_compatible", model="translator"),
                    text_reviewer=ProviderSettings(kind="openai_compatible", model="reviewer"),
                )
                resolved = resolve_provider_api_keys(options)
                self.assertEqual(resolved.asr.api_key, "cloud-asr")
                self.assertEqual(resolved.translator.api_key, "cloud-translator")
                self.assertEqual(resolved.text_reviewer.api_key, "cloud-reviewer")

    def test_gpt4o_transcribe_uses_one_vad_window_per_request(self):
        windows = [
            {"start": 1.0, "end": 3.0, "speech_scores": [0.9], "nonlexical_scores": [0.1]},
            {"start": 5.0, "end": 7.0, "speech_scores": [0.8], "nonlexical_scores": [0.1]},
        ]
        self.assertFalse(supports_segment_timestamps("gpt-4o-transcribe"))
        packs = _remote_packs(windows, timestamped=False)
        self.assertEqual(len(packs), 2)
        self.assertTrue(all(len(pack["mappings"]) == 1 for pack in packs))


if __name__ == "__main__":
    unittest.main()
