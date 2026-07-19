import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from studio.cloud_worker import CloudWhisperWorker, CloudWorkerError, _validated
from studio.main import app
from studio.runner import JobState
from studio.schemas import CloudWorkerSettings, JobOptions
from studio.settings_store import load_provider_settings, save_provider_settings


class CloudWorkerTests(unittest.TestCase):
    def test_verified_audio_upload_uses_size_and_sha256_then_reuses_file(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "audio.flac"
            audio.write_bytes(b"verified-audio" * 1024)
            worker = object.__new__(CloudWhisperWorker)
            worker.remote_job_dir = "/root/subtitle-worker/jobs/test"
            worker.checkpoint = lambda: None
            worker.logger = MagicMock()
            size, digest = worker._local_file_info(audio)
            worker._remote_file_info = MagicMock(side_effect=[None, (size, digest)])
            worker._upload_resumable = MagicMock()
            worker._exec = MagicMock(return_value="")

            result = worker._ensure_verified_audio(audio)

            self.assertEqual(result["size"], size)
            self.assertEqual(result["sha256"], digest)
            self.assertFalse(result["reused"])
            worker._upload_resumable.assert_called_once()
            publish_command = worker._exec.call_args_list[-1].args[0]
            self.assertIn("audio.flac.uploading", publish_command)
            self.assertIn("audio.ready.json", publish_command)

            worker._remote_file_info = MagicMock(return_value=(size, digest))
            worker._upload_resumable.reset_mock()
            reused = worker._ensure_verified_audio(audio)
            self.assertTrue(reused["reused"])
            worker._upload_resumable.assert_not_called()

    def test_interrupted_audio_upload_reconnects_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "audio.flac"
            audio.write_bytes(b"resumable-audio" * 1024)
            worker = object.__new__(CloudWhisperWorker)
            worker.remote_job_dir = "/root/subtitle-worker/jobs/test"
            worker.checkpoint = lambda: None
            worker.logger = MagicMock()
            size, digest = worker._local_file_info(audio)
            worker._remote_file_info = MagicMock(side_effect=[None, (size, digest)])
            worker._upload_resumable = MagicMock(side_effect=[EOFError(), None])
            worker._reconnect = MagicMock()
            worker._exec = MagicMock(return_value="")

            result = worker._ensure_verified_audio(audio)

            self.assertEqual(result["sha256"], digest)
            self.assertEqual(worker._upload_resumable.call_count, 2)
            worker._reconnect.assert_called_once()

    def test_corrupt_audio_upload_retries_three_times_and_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "audio.flac"
            audio.write_bytes(b"local-audio")
            worker = object.__new__(CloudWhisperWorker)
            worker.remote_job_dir = "/root/subtitle-worker/jobs/test"
            worker.checkpoint = lambda: None
            worker.logger = MagicMock()
            worker._remote_file_info = MagicMock(
                side_effect=[None, (1, "0" * 64), (1, "1" * 64), (1, "2" * 64)]
            )
            worker._upload_resumable = MagicMock()
            worker._exec = MagicMock(return_value="")

            with self.assertRaises(CloudWorkerError):
                worker._ensure_verified_audio(audio)

            self.assertEqual(worker._upload_resumable.call_count, 3)

    def test_controllable_remote_command_waits_for_setsid_child(self):
        class FakeChannel:
            command = ""

            def exec_command(self, command):
                self.command = command

            def exit_status_ready(self):
                return True

            def recv_ready(self):
                return False

            def recv_stderr_ready(self):
                return False

            def recv_exit_status(self):
                return 0

            def close(self):
                pass

        channel = FakeChannel()
        transport = type("Transport", (), {"open_session": lambda self: channel})()
        client = type("Client", (), {"get_transport": lambda self: transport})()
        worker = object.__new__(CloudWhisperWorker)
        worker.client = client
        worker.remote_job_dir = "/root/subtitle-worker/jobs/test"
        worker.active_control_file = ""
        worker.logger = lambda _message: None
        worker.checkpoint = lambda: None

        worker._exec("python task.py", controllable=True)

        self.assertIn("setsid --wait bash -lc", channel.command)

    def test_missing_remote_result_has_a_clear_worker_error(self):
        worker = object.__new__(CloudWhisperWorker)
        worker.sftp = type(
            "MissingSftp",
            (),
            {"get": lambda self, remote, local: (_ for _ in ()).throw(FileNotFoundError())},
        )()
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(
            CloudWorkerError
        ) as raised:
            worker._download("/remote/events.json", Path(folder) / "events.json")
        self.assertIn("没有生成预期结果文件", str(raised.exception))

    def test_worker_validation_requires_safe_connection_settings(self):
        valid = _validated(
            CloudWorkerSettings(
                enabled=True,
                host="gpu.example.com",
                port=2222,
                username="root",
                password="secret",
                remote_dir="/root/subtitle-worker",
            )
        )
        self.assertEqual(valid.port, 2222)
        with self.assertRaises(CloudWorkerError):
            _validated(
                CloudWorkerSettings(
                    enabled=True,
                    host="gpu.example.com; reboot",
                    username="root",
                    password="secret",
                )
            )

    def test_worker_credentials_never_enter_task_status(self):
        worker = CloudWorkerSettings(
            enabled=True,
            host="private-host.example",
            username="private-user",
            password="private-password",
        )
        public = JobState(
            id="cloud-job",
            options=JobOptions(input_path="movie.mp4"),
            cloud_worker_settings=worker,
        ).public()
        serialized = str(public)
        self.assertNotIn("private-host", serialized)
        self.assertNotIn("private-user", serialized)
        self.assertNotIn("private-password", serialized)

    @unittest.skipUnless(os.name == "nt", "DPAPI is available on Windows only")
    def test_worker_password_is_dpapi_encrypted_in_local_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "provider_settings.json"
            with patch("studio.settings_store.SETTINGS_PATH", path):
                save_provider_settings(
                    {
                        "cloud_worker": {
                            "enabled": True,
                            "host": "gpu.example.com",
                            "port": 22,
                            "username": "root",
                            "password": "worker-secret",
                            "remote_dir": "/root/subtitle-worker",
                        }
                    }
                )
                self.assertNotIn("worker-secret", path.read_text(encoding="utf-8"))
                loaded = load_provider_settings()
                self.assertEqual(loaded["cloud_worker"]["password"], "worker-secret")

    def test_cloud_worker_routes_are_registered(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/cloud-worker/test", paths)
        self.assertIn("/api/cloud-worker/bootstrap", paths)


if __name__ == "__main__":
    unittest.main()
