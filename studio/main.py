from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .providers import (
    OllamaProvider,
    OpenAICompatibleProvider,
    cached_whisper_models,
    whisper_model_catalog,
)
from .runner import UPLOADS_DIR, manager
from .schemas import JobOptions, ModelListRequest


APP_DIR = Path(__file__).resolve().parent
app = FastAPI(title="字幕翻译工作室", version="1.1.0")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


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
    }


@app.get("/api/models/local")
def local_models():
    ollama = []
    ollama_error = ""
    try:
        ollama = OllamaProvider().list_models()
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
    provider = request.provider
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
    path = Path(options.input_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=400, detail="输入视频不存在或不是文件")
    return manager.create(options).public()


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": manager.list()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job.public()


@app.delete("/api/jobs/{job_id}")
def archive_job(job_id: str):
    try:
        archived_to = manager.archive(job_id)
        return {"ok": True, "archived_to": str(archived_to)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/download/{output_key}")
def download(job_id: str, output_key: str):
    job = manager.get(job_id)
    if not job or output_key not in job.outputs:
        raise HTTPException(status_code=404, detail="产物不存在")
    path = Path(job.outputs[output_key])
    if not path.exists():
        raise HTTPException(status_code=404, detail="产物文件已不存在")
    return FileResponse(path, filename=path.name)
