import tempfile
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from studio.main import _bulk_job_action, _resume_paused_job, app
from studio.providers import supports_segment_timestamps
from studio.remote_asr import _remote_packs
from studio.runner import JobControl, JobManager, JobState, WeightedGpuScheduler
from studio.providers import _json_from_text
from studio.schemas import CloudWorkerSettings, JobOptions, ProviderSettings
from studio.settings_store import (
    load_provider_settings,
    resolve_provider_api_keys,
    save_provider_settings,
)


class TaskManagementTests(unittest.TestCase):
    def test_weighted_gpu_scheduler_admits_next_job_after_one_model_finishes(self):
        scheduler = WeightedGpuScheduler(capacity=5)
        self.assertTrue(scheduler.acquire("job-a", 2, timeout=0))
        self.assertTrue(scheduler.acquire("job-b", 2, timeout=0))
        self.assertFalse(scheduler.acquire("job-c", 2, timeout=0))
        self.assertEqual(scheduler.used, 4)
        self.assertEqual(scheduler.set_weight("job-a", 1), 1)
        self.assertTrue(scheduler.acquire("job-c", 2, timeout=0))
        self.assertEqual(scheduler.used, 5)
        self.assertEqual(scheduler.release("job-b"), 2)
        self.assertEqual(scheduler.used, 3)

    def test_cloud_model_completion_releases_one_dynamic_gpu_unit(self):
        scheduler = WeightedGpuScheduler(capacity=5)
        self.assertTrue(scheduler.acquire("cloud-job", 2, timeout=0))
        manager = JobManager.__new__(JobManager)
        job = JobState(id="cloud-job", options=JobOptions(input_path="movie.mp4"))
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("studio.runner.JOBS_DIR", Path(folder)),
            patch("studio.runner.REMOTE_GPU_LOCK", scheduler),
        ):
            manager.cloud_log(job, "Cohere reviewed=42")
            manager.cloud_log(job, "Qwen3-ASR windows=42 low_confidence=3")
        self.assertEqual(scheduler.used, 1)
        self.assertTrue(any("释放 1 个 GPU 动态额度" in line for line in job.logs))

    def test_individual_resume_recovers_paused_job_after_restart(self):
        job = MagicMock()
        job.options = JobOptions(input_path="movie.mp4")
        fake_manager = MagicMock()
        fake_manager.get.return_value = job
        fake_manager.controls = {}
        fake_manager.recover_paused.return_value = job
        with (
            patch("studio.main.manager", fake_manager),
            patch(
                "studio.main.load_provider_settings",
                return_value={"cloud_worker": {"enabled": True, "host": "gpu.example.com"}},
            ),
        ):
            self.assertIs(_resume_paused_job("paused-job"), job)
        fake_manager.recover_paused.assert_called_once()

    def test_cloud_model_progress_is_structured_and_persisted(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = JobManager.__new__(JobManager)
            job = JobState(id="progress-job", options=JobOptions(input_path="movie.mp4"))
            with patch("studio.runner.JOBS_DIR", Path(folder)):
                manager.update(job, log="Qwen3-ASR 8/20")
                manager.update(job, log="Cohere review 20/20")
            self.assertEqual(job.model_progress["qwen"]["current"], 8)
            self.assertEqual(job.model_progress["cohere"]["current"], 20)
            self.assertEqual(job.public()["model_progress"]["cohere"]["total"], 20)

    def test_provider_json_parser_accepts_extra_text_and_multiple_objects(self):
        self.assertEqual(
            _json_from_text('说明：```json\n{"zh":"第一句"}\n``` 完成'),
            {"zh": "第一句"},
        )
        self.assertEqual(
            _json_from_text('{"zh":"第一句"}{"zh":"多余内容"}'),
            {"zh": "第一句"},
        )

    def test_staged_job_can_be_started_with_fresh_cloud_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("studio.runner.JOBS_DIR", root / "jobs"), patch(
                "studio.runner.UPLOADS_DIR", root / "uploads"
            ):
                manager = JobManager()
                job = JobState(
                    id="staged-job",
                    options=JobOptions(input_path="movie.mp4", cloud_stage_only=True),
                    status="staged",
                    stage="音轨已校验，等待 GPU 开机",
                    progress=0.2,
                )
                manager.jobs[job.id] = job
                finished = threading.Event()
                with patch.object(manager, "_run_guarded", side_effect=lambda _job: finished.set()):
                    started = manager.start_staged(
                        job.id,
                        CloudWorkerSettings(
                            enabled=True,
                            host="gpu.example.com",
                            password="secret",
                        ),
                    )
                    self.assertTrue(finished.wait(2))

                self.assertEqual(started.status, "queued")
                self.assertFalse(started.options.cloud_stage_only)
                self.assertEqual(started.cloud_worker_settings.host, "gpu.example.com")

    def test_failed_preupload_can_be_retried_without_new_job_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("studio.runner.JOBS_DIR", root / "jobs"), patch(
                "studio.runner.UPLOADS_DIR", root / "uploads"
            ):
                manager = JobManager()
                job = JobState(
                    id="failed-stage-job",
                    options=JobOptions(input_path="movie.mp4", cloud_stage_only=True),
                    status="failed",
                    stage="处理失败",
                    error="EOFError",
                )
                manager.jobs[job.id] = job
                finished = threading.Event()
                with patch.object(manager, "_run_guarded", side_effect=lambda _job: finished.set()):
                    retried = manager.retry_staged_upload(
                        job.id,
                        CloudWorkerSettings(
                            enabled=True,
                            host="gpu.example.com",
                            password="secret",
                        ),
                    )
                    self.assertTrue(finished.wait(2))

                self.assertEqual(retried.id, "failed-stage-job")
                self.assertEqual(retried.status, "queued")
                self.assertTrue(retried.options.cloud_stage_only)
                self.assertEqual(retried.error, "")

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

    def test_persisted_paused_job_can_be_canceled_without_live_control(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("studio.runner.JOBS_DIR", root / "jobs"), patch(
                "studio.runner.UPLOADS_DIR", root / "uploads"
            ):
                manager = JobManager()
                job = JobState(
                    id="persisted-paused",
                    options=JobOptions(input_path="movie.mp4", cloud_stage_only=True),
                    status="paused",
                    stage="已暂停 · 等待云节点",
                )
                manager.jobs[job.id] = job

                canceled = manager.cancel(job.id)

                self.assertEqual(canceled.status, "canceled")
                self.assertNotIn(job.id, manager.controls)
                self.assertIn("仅保留日志", canceled.logs[-1])

    def test_cancel_prunes_task_files_but_keeps_log_record_and_source_video(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = root / "jobs"
            uploads = root / "uploads"
            source = root / "source.mp4"
            output = root / "movie_zh.srt"
            source.write_bytes(b"source-video")
            output.write_text("subtitle", encoding="utf-8")
            with patch("studio.runner.JOBS_DIR", jobs), patch(
                "studio.runner.UPLOADS_DIR", uploads
            ):
                manager = JobManager()
                job = JobState(
                    id="cancel-keep-log",
                    options=JobOptions(input_path=str(source), cloud_stage_only=True),
                    status="staged",
                    outputs={"srt": str(output)},
                )
                manager.jobs[job.id] = job
                work = jobs / job.id / "work"
                work.mkdir(parents=True)
                (work / "cloud_audio.flac").write_bytes(b"audio")
                manager.persist(job)

                manager.cancel(job.id)
                manager.prune_canceled_to_logs(job.id)

                self.assertTrue(source.exists())
                self.assertFalse(output.exists())
                self.assertFalse(work.exists())
                self.assertEqual(
                    {path.name for path in (jobs / job.id).iterdir()}, {"status.json"}
                )
                restored = JobManager().get(job.id)
                self.assertEqual(restored.status, "canceled")
                self.assertEqual(restored.outputs, {})
                self.assertTrue(restored.logs)

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

    def test_paused_job_survives_restart_and_output_settings_can_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = root / "jobs"
            uploads = root / "uploads"
            with patch("studio.runner.JOBS_DIR", jobs), patch(
                "studio.runner.UPLOADS_DIR", uploads
            ):
                manager = JobManager()
                job = JobState(
                    id="paused-job",
                    options=JobOptions(
                        input_path="movie.mp4",
                        create_soft_subtitle_video=True,
                    ),
                    status="paused",
                    stage="已暂停 · 日语语音识别",
                    progress=0.57,
                )
                manager.jobs[job.id] = job
                manager.persist(job)

                restored = JobManager()
                loaded = restored.get(job.id)
                self.assertEqual(loaded.status, "paused")
                self.assertNotIn(job.id, restored.controls)
                restored.apply_output_settings(
                    job.id,
                    create_soft_subtitle_video=False,
                    create_hard_subtitle_video=False,
                )
                self.assertFalse(loaded.options.create_soft_subtitle_video)
                self.assertFalse(loaded.options.create_hard_subtitle_video)

    def test_paused_job_only_allows_unlocked_stage_settings_to_change(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "studio.runner.JOBS_DIR", Path(folder) / "jobs"
        ), patch("studio.runner.UPLOADS_DIR", Path(folder) / "uploads"):
            manager = JobManager()
            job = JobState(
                id="stage-lock-job",
                options=JobOptions(input_path="movie.mp4", profile="balanced"),
                status="paused",
                locked_config_groups=["recognition"],
            )
            manager.jobs[job.id] = job
            manager.update_paused_settings(
                job.id, create_soft_subtitle_video=False
            )
            self.assertFalse(job.options.create_soft_subtitle_video)
            with self.assertRaisesRegex(RuntimeError, "识别策略阶段已经开始"):
                manager.update_paused_settings(job.id, profile="recall")
            job.locked_config_groups.append("output")
            with self.assertRaisesRegex(RuntimeError, "字幕与视频产物阶段已经开始"):
                manager.update_paused_settings(
                    job.id, create_hard_subtitle_video=True
                )

    def test_public_job_exposes_stage_locks_without_secrets(self):
        job = JobState(
            id="public-lock-job",
            options=JobOptions(input_path="movie.mp4"),
            locked_config_groups=["recognition", "translation"],
        )
        public = job.public()
        self.assertEqual(public["locked_config_groups"], ["recognition", "translation"])
        self.assertEqual(public["config_group_labels"]["output"], "字幕与视频产物")

    def test_task_control_routes_are_registered(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/jobs/{job_id}/pause", paths)
        self.assertIn("/api/jobs/{job_id}/resume", paths)
        self.assertIn("/api/jobs/{job_id}/settings", paths)
        self.assertIn("/api/jobs/{job_id}/cancel", paths)
        self.assertIn("/api/jobs/actions/pause-all", paths)
        self.assertIn("/api/jobs/actions/resume-all", paths)
        self.assertIn("/api/jobs/actions/cancel-all", paths)
        self.assertIn("/api/jobs/actions/delete-finished", paths)
        self.assertIn("/api/jobs/actions/delete-all", paths)
        self.assertIn("/api/jobs/actions/start-staged-all", paths)
        self.assertIn("/api/jobs/actions/retry-stage-failed", paths)
        self.assertIn("/api/jobs/actions/retry-failed-all", paths)
        self.assertIn("/api/jobs/{job_id}/start-staged", paths)
        self.assertIn("/api/jobs/{job_id}/retry-stage", paths)

    def test_bulk_actions_only_target_eligible_job_states(self):
        class FakeManager:
            def __init__(self):
                self.paused = []
                self.deleted = []

            def list(self):
                return [
                    {"id": "running", "status": "running"},
                    {"id": "queued", "status": "queued"},
                    {"id": "done", "status": "completed"},
                ]

            def pause(self, job_id):
                self.paused.append(job_id)

            def delete(self, job_id):
                self.deleted.append(job_id)

            def get(self, job_id):
                return None

        fake = FakeManager()
        with patch("studio.main.manager", fake), patch(
            "studio.main._delete_job_completely",
            side_effect=lambda job_id: (fake.delete(job_id), ""),
        ):
            paused = _bulk_job_action("pause", {"queued", "running"})
            deleted = _bulk_job_action("delete", {"completed", "failed", "canceled"})

        self.assertEqual(paused["count"], 2)
        self.assertEqual(set(fake.paused), {"running", "queued"})
        self.assertEqual(deleted["count"], 1)
        self.assertEqual(fake.deleted, ["done"])

    def test_task_panel_exposes_total_progress_and_explicit_bulk_actions(self):
        html = (Path(__file__).parents[1] / "studio" / "static" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (Path(__file__).parents[1] / "studio" / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="overall-progress-bar"', html)
        self.assertIn('id="stage-cloud"', html)
        self.assertIn('id="retry-all-failed-jobs"', html)
        self.assertIn('id="delete-all-jobs"', html)
        self.assertIn("支持无卡", html)
        self.assertIn('id="cloud-setup-logs"', html)
        self.assertIn("monitorCloudSetup", script)
        self.assertIn("renderOverallProgress(data.jobs)", script)
        self.assertIn("taskDisplayState", script)
        self.assertIn("排队等待 GPU", script)
        self.assertIn("cloud_gpu_job_concurrency", script)
        self.assertIn("retry-failed-all", script)
        self.assertIn("delete-all", script)

    def test_update_maintenance_lock_blocks_all_program_operations(self):
        source = (Path(__file__).parents[1] / "studio" / "main.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('MAINTENANCE_LOCK = DATA_DIR / "maintenance.lock"', source)
        self.assertIn("if MAINTENANCE_LOCK.exists()", source)
        self.assertIn("status_code=503", source)

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

    def test_blank_secret_fields_preserve_encrypted_local_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "provider_settings.json"
            with patch("studio.settings_store.SETTINGS_PATH", path):
                first = {
                    "translator": {
                        "kind": "openai_compatible",
                        "base_url": "https://example.com/v1",
                        "api_key": "keep-this-key",
                        "model": "old-model",
                    },
                    "cloud_worker": {
                        "password": "keep-this-password",
                        "huggingface_token": "keep-this-hf-token",
                    },
                }
                save_provider_settings(first)
                save_provider_settings(
                    {
                        "translator": {
                            "kind": "openai_compatible",
                            "base_url": "https://example.com/v1",
                            "api_key": "",
                            "model": "new-model",
                        },
                        "cloud_worker": {"password": "", "huggingface_token": ""},
                    }
                )
                loaded = load_provider_settings(expose_secrets=True)
                self.assertEqual(loaded["translator"]["api_key"], "keep-this-key")
                self.assertEqual(loaded["translator"]["model"], "new-model")
                self.assertEqual(
                    loaded["cloud_worker"]["password"], "keep-this-password"
                )
                self.assertEqual(
                    loaded["cloud_worker"]["huggingface_token"],
                    "keep-this-hf-token",
                )
                raw = path.read_text(encoding="utf-8")
                self.assertNotIn("keep-this-hf-token", raw)

    def test_resolve_provider_keys_falls_back_to_encrypted_local_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "provider_settings.json"
            with patch("studio.settings_store.SETTINGS_PATH", path), patch.dict(
                "os.environ", {"SUBTITLE_TRANSLATOR_API_KEY": ""}, clear=False
            ):
                save_provider_settings(
                    {"translator": {"api_key": "saved-translator-key"}}
                )
                options = JobOptions(input_path="movie.mp4")
                resolved = resolve_provider_api_keys(options)
                self.assertEqual(
                    resolved.translator.api_key, "saved-translator-key"
                )

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
