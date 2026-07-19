import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from studio.main import _choose_local_folder, _video_files_in, app, create_folder_jobs
from studio.schemas import FolderBatchRequest, JobOptions


class BatchFolderTests(unittest.TestCase):
    def test_folder_scan_reads_supported_top_level_videos_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "B.MKV").write_bytes(b"")
            (root / "a.mp4").write_bytes(b"")
            (root / "note.txt").write_text("ignore", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "hidden.mp4").write_bytes(b"")
            resolved, files = _video_files_in(str(root))
            self.assertEqual(resolved, root.resolve())
            self.assertEqual([path.name for path in files], ["a.mp4", "B.MKV"])

    def test_batch_creates_one_job_per_video_with_shared_output_dir(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "one.mp4").write_bytes(b"")
            (source / "one.mkv").write_bytes(b"")
            (source / "two.mov").write_bytes(b"")
            captured = []

            def fake_create(options, worker):
                captured.append(options)
                return SimpleNamespace(public=lambda: {"input": options.input_path})

            request = FolderBatchRequest(
                input_dir=str(source),
                output_dir=str(output),
                options=JobOptions(input_path=""),
            )
            with patch("studio.main.manager.create", side_effect=fake_create), patch(
                "studio.main.load_provider_settings", return_value={"cloud_worker": {}}
            ):
                result = create_folder_jobs(request)
            self.assertEqual(result["count"], 3)
            self.assertTrue(output.is_dir())
            self.assertTrue(all(item.output_dir == str(output.resolve()) for item in captured))
            self.assertEqual(
                [Path(item.input_path).name for item in captured],
                ["one.mkv", "one.mp4", "two.mov"],
            )
            self.assertEqual(
                [item.output_name for item in captured],
                ["one", "one_mp4", "two"],
            )

    def test_batch_rejects_output_equal_to_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "one.mp4").write_bytes(b"")
            request = FolderBatchRequest(
                input_dir=str(root),
                output_dir=str(root),
                options=JobOptions(input_path=""),
            )
            with self.assertRaises(HTTPException) as raised:
                create_folder_jobs(request)
            self.assertEqual(raised.exception.status_code, 400)

    def test_batch_routes_are_registered(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/media/folder", paths)
        self.assertIn("/api/jobs/batch", paths)
        self.assertIn("/api/local/pick-folder", paths)

    def test_folder_picker_is_blocked_outside_local_mode(self):
        with patch("studio.main.ALLOW_LOCAL_OPEN", False), self.assertRaises(
            HTTPException
        ) as raised:
            _choose_local_folder()
        self.assertEqual(raised.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
