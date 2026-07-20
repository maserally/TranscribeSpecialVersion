from __future__ import annotations

import base64
import binascii
import os
import posixpath
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .cloud_worker import CloudWhisperWorker
from .config import (
    ALLOW_LOCAL_OPEN,
    BASIC_AUTH_ENABLED,
    BASIC_AUTH_PASSWORD,
    BASIC_AUTH_USERNAME,
    CLOUD_MODE,
    DATA_DIR,
    PROVIDER_KEY_ENV,
    SECRET_POLICY,
)
from .providers import (
    OllamaProvider,
    OpenAICompatibleProvider,
    cached_whisper_models,
    whisper_model_catalog,
)
from .runner import JOBS_DIR, UPLOADS_DIR, manager
from .schemas import (
    CloudWorkerRequest,
    CloudWorkerSettings,
    FolderBatchRequest,
    FolderPickerRequest,
    FolderScanRequest,
    JobOptions,
    JobOutputSettings,
    ModelListRequest,
    SavedProviderSettings,
)
from .settings_store import (
    load_provider_settings,
    resolve_provider_api_keys,
    save_provider_settings,
)


APP_DIR = Path(__file__).resolve().parent
app = FastAPI(title="字幕翻译工作室", version="1.13.6")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".ts", ".m2ts", ".wmv", ".flv",
}
MAX_BATCH_FILES = 500


@app.middleware("http")
async def optional_basic_auth(request: Request, call_next):
    if not BASIC_AUTH_ENABLED:
        return await call_next(request)
    authorization = request.headers.get("Authorization", "")
    valid = False
    if authorization.startswith("Basic "):
        try:
            decoded = base64.b64decode(authorization[6:]).decode("utf-8")
            username, password = decoded.split(":", 1)
            valid = secrets.compare_digest(username, BASIC_AUTH_USERNAME) and secrets.compare_digest(
                password, BASIC_AUTH_PASSWORD
            )
        except (ValueError, UnicodeDecodeError, binascii.Error):
            pass
    if not valid:
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Subtitle Studio"'})
    return await call_next(request)


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    gpu = ""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpu = result.stdout.strip()
    except Exception:
        pass
    return {
        "ok": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
        "gpu": gpu,
        "mode": "cloud" if CLOUD_MODE else "local",
    }


@app.get("/api/runtime")
def runtime():
    return {
        "mode": "cloud" if CLOUD_MODE else "local",
        "local_open": ALLOW_LOCAL_OPEN,
        "local_path_input": not CLOUD_MODE,
        "download_outputs": True,
        "data_dir": str(DATA_DIR),
        "auth_enabled": BASIC_AUTH_ENABLED,
        "secret_policy": SECRET_POLICY,
    }


@app.get("/api/models/local")
def local_models():
    ollama = []
    ollama_error = ""
    try:
        ollama = OllamaProvider(timeout=3).list_models()
    except Exception as exc:
        ollama_error = str(exc)
    return {
        "whisper": [item["id"] for item in whisper_model_catalog()],
        "whisper_catalog": whisper_model_catalog(),
        "whisper_installed": cached_whisper_models(),
        "ollama": ollama,
        "ollama_error": ollama_error,
    }


@app.post("/api/models")
def provider_models(request: ModelListRequest):
    provider = request.provider.model_copy(deep=True)
    if not provider.api_key and request.role:
        provider.api_key = os.getenv(PROVIDER_KEY_ENV[request.role], "")
    try:
        if provider.kind == "local_ollama":
            models = OllamaProvider(provider.base_url or "http://127.0.0.1:11434").list_models()
        elif provider.kind == "openai_compatible":
            models = OpenAICompatibleProvider(provider.base_url, provider.api_key).list_models()
        elif provider.kind == "local_whisper":
            models = [item["id"] for item in whisper_model_catalog()]
        else:
            raise ValueError("不支持的模型提供方")
        return {"models": models}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/settings/providers")
def get_provider_settings():
    return load_provider_settings(expose_secrets=not CLOUD_MODE)


@app.put("/api/settings/providers")
def put_provider_settings(settings: SavedProviderSettings):
    path = save_provider_settings(settings.model_dump())
    return {
        "ok": True,
        "path": str(path),
        "secret_policy": SECRET_POLICY,
    }


@app.post("/api/uploads")
async def upload(file: UploadFile = File(...)):
    safe_name = re.sub(r"[^\w.()\-\u4e00-\u9fff]+", "_", file.filename or "video.mp4")
    upload_id = __import__("uuid").uuid4().hex[:12]
    target_dir = UPLOADS_DIR / upload_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / safe_name
    size = 0
    async with aiofiles.open(target, "wb") as output:
        while chunk := await file.read(8 * 1024 * 1024):
            size += len(chunk)
            await output.write(chunk)
    return {"path": str(target), "size": size, "name": safe_name}


@app.post("/api/jobs")
def create_job(options: JobOptions):
    options = resolve_provider_api_keys(options)
    path = Path(options.input_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=400, detail="输入视频不存在或不是文件")
    if options.output_dir:
        output_dir = Path(options.output_dir).expanduser().resolve()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"无法创建输出目录：{exc}") from exc
        options.output_dir = str(output_dir)
    saved = load_provider_settings(expose_secrets=True)
    worker = CloudWorkerSettings.model_validate(saved.get("cloud_worker", {}))
    if options.cloud_stage_only and (
        not worker.enabled or options.asr.kind != "local_whisper"
    ):
        raise HTTPException(
            status_code=400,
            detail="无卡预上传必须启用云 GPU 运算单元，并选择本地 Whisper 识别",
        )
    return manager.create(options, worker if worker.enabled else None).public()


def _video_files_in(folder_text: str, recursive: bool = True) -> tuple[Path, list[Path]]:
    folder = Path(folder_text).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="输入文件夹不存在或不是文件夹")
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    files = sorted(
        (
            path.resolve()
            for path in candidates
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and path.resolve().is_relative_to(folder)
        ),
        key=lambda path: str(path.relative_to(folder)).casefold(),
    )
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"文件夹包含 {len(files)} 个视频，单次最多创建 {MAX_BATCH_FILES} 个任务",
        )
    return folder, files


@app.post("/api/media/folder")
def scan_media_folder(request: FolderScanRequest):
    folder, files = _video_files_in(request.input_dir, request.recursive)
    return {
        "folder": str(folder),
        "count": len(files),
        "recursive": request.recursive,
        "files": [
            {
                "name": path.name,
                "relative_path": str(path.relative_to(folder)),
                "path": str(path),
            }
            for path in files
        ],
    }


def _choose_local_folder(initial_dir: str = "", title: str = "选择文件夹") -> str:
    if not ALLOW_LOCAL_OPEN:
        raise HTTPException(status_code=409, detail="当前运行模式不能打开服务器端文件夹选择器")
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        try:
            root.withdraw()
            root.attributes("-topmost", True)
            root.update()
            initial = Path(initial_dir).expanduser() if initial_dir else None
            selected = filedialog.askdirectory(
                parent=root,
                title=title[:60] or "选择文件夹",
                initialdir=str(initial) if initial and initial.is_dir() else None,
                mustexist=False,
            )
        finally:
            root.destroy()
        return str(Path(selected).resolve()) if selected else ""
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"无法打开系统文件夹选择器：{exc}") from exc


@app.post("/api/local/pick-folder")
def pick_local_folder(request: FolderPickerRequest):
    selected = _choose_local_folder(request.initial_dir, request.title)
    return {"path": selected, "cancelled": not bool(selected)}


@app.post("/api/jobs/batch")
def create_folder_jobs(request: FolderBatchRequest):
    folder, files = _video_files_in(request.input_dir, request.recursive)
    if request.selected_files is not None:
        discovered = {os.path.normcase(str(path)): path for path in files}
        requested: set[str] = set()
        invalid: list[str] = []
        for selected in request.selected_files:
            relative = Path(str(selected).strip())
            candidate = (folder / relative).resolve()
            key = os.path.normcase(str(candidate))
            if (
                not str(selected).strip()
                or relative.is_absolute()
                or not candidate.is_relative_to(folder)
                or key not in discovered
            ):
                invalid.append(str(selected))
            else:
                requested.add(key)
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"所选文件不在本次扫描结果中：{invalid[0]}",
            )
        files = [path for path in files if os.path.normcase(str(path)) in requested]
    if not files:
        if request.selected_files is not None:
            raise HTTPException(status_code=400, detail="请至少选择一个需要转换的视频")
        scope = "及其子文件夹" if request.recursive else "第一层"
        raise HTTPException(status_code=400, detail=f"该文件夹{scope}没有支持的视频文件")
    output_dir = None
    if request.output_dir:
        output_dir = Path(request.output_dir).expanduser().resolve()
        if output_dir == folder or output_dir.is_relative_to(folder):
            raise HTTPException(status_code=400, detail="批量输出目录不能位于输入目录内部")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"无法创建输出目录：{exc}") from exc
    saved = load_provider_settings(expose_secrets=True)
    worker = CloudWorkerSettings.model_validate(saved.get("cloud_worker", {}))
    base_options = resolve_provider_api_keys(request.options)
    if base_options.cloud_stage_only and (
        not worker.enabled or base_options.asr.kind != "local_whisper"
    ):
        raise HTTPException(
            status_code=400,
            detail="无卡预上传必须启用云 GPU 运算单元，并选择本地 Whisper 识别",
        )
    created = []
    used_output_names: dict[str, set[str]] = {}
    for media_path in files:
        options = base_options.model_copy(deep=True)
        options.input_path = str(media_path)
        relative_parent = media_path.relative_to(folder).parent
        job_output_dir = output_dir / relative_parent if output_dir else None
        name_scope = str(relative_parent).casefold()
        scoped_names = used_output_names.setdefault(name_scope, set())
        output_name = media_path.stem
        if output_name.casefold() in scoped_names:
            output_name = f"{media_path.stem}_{media_path.suffix.lstrip('.').lower()}"
        suffix_index = 2
        base_output_name = output_name
        while output_name.casefold() in scoped_names:
            output_name = f"{base_output_name}_{suffix_index}"
            suffix_index += 1
        scoped_names.add(output_name.casefold())
        options.output_name = output_name
        options.output_dir = str(job_output_dir) if job_output_dir else ""
        created.append(manager.create(options, worker if worker.enabled else None).public())
    return {"count": len(created), "folder": str(folder), "jobs": created}


@app.post("/api/cloud-worker/test")
def test_cloud_worker(request: CloudWorkerRequest):
    try:
        return {"ok": True, **CloudWhisperWorker(request.cloud_worker).test_connection()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/cloud-worker/bootstrap")
def bootstrap_cloud_worker(request: CloudWorkerRequest):
    worker = CloudWhisperWorker(request.cloud_worker)
    try:
        worker.connect()
        result = worker.bootstrap()
        return {"ok": True, **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        worker.close()


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": manager.list()}


def _cleanup_preuploaded_audio(job):
    if not job or not (job.status == "staged" or job.options.cloud_stage_only):
        return
    saved = load_provider_settings(expose_secrets=True)
    worker_settings = job.cloud_worker_settings or CloudWorkerSettings.model_validate(
        saved.get("cloud_worker", {})
    )
    worker = CloudWhisperWorker(worker_settings)
    try:
        worker.connect()
        worker.remote_job_dir = posixpath.join(worker.settings.remote_dir, "jobs", job.id)
        worker.cleanup_job()
    finally:
        worker.close()


def _bulk_job_action(action: str, eligible_statuses: set[str]):
    jobs = manager.list()
    targets = [job["id"] for job in jobs if job["status"] in eligible_statuses]
    succeeded = []
    failed = []
    for job_id in targets:
        try:
            if action == "delete":
                _cleanup_preuploaded_audio(manager.get(job_id))
                manager.delete(job_id)
            else:
                getattr(manager, action)(job_id)
            succeeded.append(job_id)
        except Exception as exc:
            failed.append({"id": job_id, "error": str(exc)})
    return {
        "ok": not failed,
        "action": action,
        "count": len(succeeded),
        "succeeded": succeeded,
        "failed": failed,
    }


@app.post("/api/jobs/actions/pause-all")
def pause_all_jobs():
    return _bulk_job_action("pause", {"queued", "running"})


@app.post("/api/jobs/actions/resume-all")
def resume_all_jobs(output_settings: JobOutputSettings):
    saved = load_provider_settings(expose_secrets=True)
    worker = CloudWorkerSettings.model_validate(saved.get("cloud_worker", {}))
    targets = [job["id"] for job in manager.list() if job["status"] == "paused"]
    succeeded = []
    failed = []
    for job_id in targets:
        try:
            job = manager.get(job_id)
            job.options = resolve_provider_api_keys(job.options)
            manager.apply_output_settings(
                job_id,
                create_soft_subtitle_video=output_settings.create_soft_subtitle_video,
                create_hard_subtitle_video=output_settings.create_hard_subtitle_video,
            )
            if job_id in manager.controls:
                manager.resume(job_id)
            else:
                manager.recover_paused(job_id, worker if worker.enabled else None)
            succeeded.append(job_id)
        except (KeyError, RuntimeError) as exc:
            failed.append({"id": job_id, "error": str(exc)})
    return {
        "ok": not failed,
        "action": "resume-all",
        "count": len(succeeded),
        "succeeded": succeeded,
        "failed": failed,
    }


@app.post("/api/jobs/actions/cancel-all")
def cancel_all_jobs():
    return _bulk_job_action("cancel", {"queued", "running", "paused"})


@app.post("/api/jobs/actions/start-staged-all")
def start_all_staged_jobs():
    saved = load_provider_settings(expose_secrets=True)
    worker = CloudWorkerSettings.model_validate(saved.get("cloud_worker", {}))
    if not worker.enabled:
        raise HTTPException(status_code=400, detail="请先启用并保存云 GPU 运算单元配置")
    targets = [job["id"] for job in manager.list() if job["status"] == "staged"]
    succeeded = []
    failed = []
    for job_id in targets:
        try:
            manager.start_staged(job_id, worker)
            succeeded.append(job_id)
        except (KeyError, RuntimeError) as exc:
            failed.append({"id": job_id, "error": str(exc)})
    return {
        "ok": not failed,
        "action": "start-staged-all",
        "count": len(succeeded),
        "succeeded": succeeded,
        "failed": failed,
    }


@app.post("/api/jobs/actions/retry-stage-failed")
def retry_failed_staged_uploads():
    saved = load_provider_settings(expose_secrets=True)
    worker = CloudWorkerSettings.model_validate(saved.get("cloud_worker", {}))
    if not worker.enabled:
        raise HTTPException(status_code=400, detail="请先启用并保存云 GPU 运算单元配置")
    targets = [
        job["id"]
        for job in manager.list()
        if job["status"] == "failed" and job["options"].get("cloud_stage_only")
    ]
    succeeded = []
    failed = []
    for job_id in targets:
        try:
            manager.retry_staged_upload(job_id, worker)
            succeeded.append(job_id)
        except (KeyError, RuntimeError) as exc:
            failed.append({"id": job_id, "error": str(exc)})
    return {
        "ok": not failed,
        "action": "retry-stage-failed",
        "count": len(succeeded),
        "succeeded": succeeded,
        "failed": failed,
    }


@app.delete("/api/jobs/actions/delete-finished")
def delete_finished_jobs():
    return _bulk_job_action("delete", {"completed", "failed", "canceled"})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job.public()


@app.post("/api/jobs/{job_id}/start-staged")
def start_staged_job(job_id: str):
    saved = load_provider_settings(expose_secrets=True)
    worker = CloudWorkerSettings.model_validate(saved.get("cloud_worker", {}))
    if not worker.enabled:
        raise HTTPException(status_code=400, detail="请先启用并保存云 GPU 运算单元配置")
    try:
        return manager.start_staged(job_id, worker).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/retry-stage")
def retry_staged_job(job_id: str):
    saved = load_provider_settings(expose_secrets=True)
    worker = CloudWorkerSettings.model_validate(saved.get("cloud_worker", {}))
    if not worker.enabled:
        raise HTTPException(status_code=400, detail="请先启用并保存云 GPU 运算单元配置")
    try:
        return manager.retry_staged_upload(job_id, worker).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    try:
        job = manager.get(job_id)
        _cleanup_preuploaded_audio(job)
        deleted = manager.delete(job_id)
        return {"ok": True, "deleted": str(deleted)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail=f"无法清理云端预上传音轨，任务尚未删除：{exc}",
        ) from exc


def _job_action(job_id: str, action: str):
    try:
        return getattr(manager, action)(job_id).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str):
    return _job_action(job_id, "pause")


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    return _job_action(job_id, "resume")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    return _job_action(job_id, "cancel")


@app.get("/api/jobs/{job_id}/download/{output_key}")
def download(job_id: str, output_key: str):
    path = _output_path(job_id, output_key)
    return FileResponse(path, filename=path.name)


def _output_path(job_id: str, output_key: str) -> Path:
    job = manager.get(job_id)
    if not job or output_key not in job.outputs:
        raise HTTPException(status_code=404, detail="产物不存在")
    path = Path(job.outputs[output_key]).resolve()
    job_root = (JOBS_DIR / job_id).resolve()
    allowed_roots = [job_root]
    configured_output = getattr(getattr(job, "options", None), "output_dir", "")
    if configured_output:
        allowed_roots.append(Path(configured_output).expanduser().resolve())
    if not any(path.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(status_code=403, detail="产物路径不在任务或指定输出目录内")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="产物文件已不存在")
    return path


def _open_local(path: Path):
    if not ALLOW_LOCAL_OPEN:
        raise HTTPException(status_code=409, detail="云算力模式不支持服务器端打开，请下载文件")
    try:
        if os.name == "nt":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"无法调用本机默认程序：{exc}") from exc


@app.post("/api/jobs/{job_id}/open/{output_key}")
def open_output(job_id: str, output_key: str):
    path = _output_path(job_id, output_key)
    _open_local(path)
    return {"ok": True, "path": str(path)}


@app.post("/api/jobs/{job_id}/open-folder")
def open_output_folder(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    configured_output = getattr(job.options, "output_dir", "")
    folder = (
        Path(configured_output).expanduser().resolve()
        if configured_output
        else (JOBS_DIR / job_id / "output").resolve()
    )
    job_root = (JOBS_DIR / job_id).resolve()
    allowed = folder.is_relative_to(job_root) or bool(configured_output)
    if not allowed or not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail="产物文件夹不存在")
    _open_local(folder)
    return {"ok": True, "path": str(folder)}
