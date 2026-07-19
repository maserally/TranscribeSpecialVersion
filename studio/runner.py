from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from .cloud_worker import CloudWhisperWorker
from .config import DATA_DIR, ROOT
from .languages import language_info
from .quality import PROFILE_SETTINGS, finalize_cues, quality_summary
from .recall import (
    accepted_recovery_rows,
    filter_events_for_uncovered_speech,
    filter_events_for_gaps,
    merge_recovery,
    save_gap_audit,
    vad_fallback_events_for_gaps,
)
from .remote_asr import run_remote_asr
from .schemas import CloudWorkerSettings, JobOptions
from .subtitles import mux_hard_subtitles, mux_soft_subtitles, write_subtitles
from .text_review import review_cues
from .translation import translate_cues
from .asr_context import attach_asr_reviews


JOBS_DIR = DATA_DIR / "jobs"
UPLOADS_DIR = DATA_DIR / "uploads"
GPU_LOCK = threading.Lock()


class JobCancelled(RuntimeError):
    pass


def _open_gate():
    gate = threading.Event()
    gate.set()
    return gate


@dataclass
class JobControl:
    run_gate: threading.Event = field(default_factory=_open_gate)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    finished_event: threading.Event = field(default_factory=threading.Event)
    process: subprocess.Popen | None = None
    previous_status: str = "queued"
    previous_stage: str = "等待处理"


def _now():
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class JobState:
    id: str
    options: JobOptions
    status: str = "queued"
    stage: str = "等待处理"
    progress: float = 0.0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    logs: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    error: str = ""
    cloud_worker_settings: CloudWorkerSettings | None = field(default=None, repr=False)
    cloud_session: Any = field(default=None, repr=False)

    def public(self):
        safe_options = self.options.model_dump()
        safe_options["asr"]["api_key"] = ""
        safe_options["translator"]["api_key"] = ""
        safe_options["text_reviewer"]["api_key"] = ""
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "logs": self.logs[-200:],
            "outputs": self.outputs,
            "error": self.error,
            "options": safe_options,
        }


class JobManager:
    def __init__(self):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, JobState] = {}
        self.controls: dict[str, JobControl] = {}
        self.lock = threading.Lock()
        self._load_existing()

    def _load_existing(self):
        for status_path in JOBS_DIR.glob("*/status.json"):
            try:
                data = json.loads(status_path.read_text(encoding="utf-8"))
                options = JobOptions.model_validate(data["options"])
                status = data.get("status", "failed")
                stage = data.get("stage", "")
                error = data.get("error", "")
                if status in {"queued", "running", "paused"}:
                    status = "failed"
                    stage = "上次运行被中断"
                    error = "软件上次关闭时任务尚未完成，请重新创建任务"
                job = JobState(
                    id=data["id"],
                    options=options,
                    status=status,
                    stage=stage,
                    progress=float(data.get("progress", 0)),
                    created_at=data.get("created_at", _now()),
                    updated_at=data.get("updated_at", _now()),
                    logs=list(data.get("logs", [])),
                    outputs=dict(data.get("outputs", {})),
                    error=error,
                )
                self.jobs[job.id] = job
            except Exception:
                continue

    def create(
        self,
        options: JobOptions,
        cloud_worker_settings: CloudWorkerSettings | None = None,
    ):
        job = JobState(
            id=uuid.uuid4().hex[:12],
            options=options,
            cloud_worker_settings=cloud_worker_settings,
        )
        with self.lock:
            self.jobs[job.id] = job
            self.controls[job.id] = JobControl()
        self.persist(job)
        threading.Thread(target=self._run_guarded, args=(job,), daemon=True).start()
        return job

    def start_staged(self, job_id: str, cloud_worker_settings: CloudWorkerSettings):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status != "staged":
                raise RuntimeError("只有已经校验并预上传完成的任务可以开始云端处理")
            job.cloud_worker_settings = cloud_worker_settings
            job.options.cloud_stage_only = False
            job.status = "queued"
            job.stage = "已校验音轨，等待 GPU 处理"
            job.error = ""
            self.controls[job.id] = JobControl()
        self.update(job, progress=max(job.progress, 0.20), log="开始使用已校验的云端音轨")
        threading.Thread(target=self._run_guarded, args=(job,), daemon=True).start()
        return job

    def retry_staged_upload(
        self, job_id: str, cloud_worker_settings: CloudWorkerSettings
    ):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status != "failed" or not job.options.cloud_stage_only:
                raise RuntimeError("只有失败的无卡预上传任务可以重试")
            job.cloud_worker_settings = cloud_worker_settings
            job.status = "queued"
            job.stage = "等待断点续传"
            job.error = ""
            self.controls[job.id] = JobControl()
        self.update(job, log="重新连接云节点，将校验已有临时分片并断点续传")
        threading.Thread(target=self._run_guarded, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str):
        return self.jobs.get(job_id)

    def list(self):
        with self.lock:
            stale = [
                job_id
                for job_id, job in self.jobs.items()
                if job.status not in {"queued", "running", "paused"}
                and not (JOBS_DIR / job_id).exists()
            ]
            for job_id in stale:
                self.jobs.pop(job_id, None)
                self.controls.pop(job_id, None)
        return sorted((x.public() for x in self.jobs.values()), key=lambda x: x["created_at"], reverse=True)

    def archive(self, job_id: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status in {"queued", "running", "paused"}:
                raise RuntimeError("运行中的任务不能移除")
            self.jobs.pop(job_id)
        source = JOBS_DIR / job_id
        archive_dir = DATA_DIR / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / f"{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if source.exists():
            source.replace(target)
        return target

    def _control(self, job_id: str) -> JobControl:
        with self.lock:
            control = self.controls.get(job_id)
        if not control:
            raise RuntimeError("该任务当前没有可控制的运行进程")
        return control

    @staticmethod
    def _processes(process: subprocess.Popen | None):
        if not process or process.poll() is not None:
            return []
        try:
            parent = psutil.Process(process.pid)
            return [*parent.children(recursive=True), parent]
        except psutil.Error:
            return []

    def pause(self, job_id: str):
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status not in {"queued", "running"}:
            raise RuntimeError("只有等待中或运行中的任务可以暂停")
        control = self._control(job_id)
        control.previous_status = job.status
        control.previous_stage = job.stage
        control.run_gate.clear()
        if job.cloud_session:
            job.cloud_session.pause_current()
        for process in self._processes(control.process):
            try:
                process.suspend()
            except psutil.Error:
                pass
        self.update(job, status="paused", stage=f"已暂停 · {control.previous_stage}", log="任务已暂停")
        return job

    def resume(self, job_id: str):
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status != "paused":
            raise RuntimeError("任务当前没有暂停")
        control = self._control(job_id)
        if job.cloud_session:
            job.cloud_session.resume_current()
        for process in reversed(self._processes(control.process)):
            try:
                process.resume()
            except psutil.Error:
                pass
        control.run_gate.set()
        self.update(
            job,
            status=control.previous_status,
            stage=control.previous_stage,
            log="任务已继续",
        )
        return job

    def cancel(self, job_id: str):
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status not in {"queued", "running", "paused"}:
            raise RuntimeError("任务已经结束")
        control = self._control(job_id)
        control.cancel_event.set()
        control.run_gate.set()
        if job.cloud_session:
            job.cloud_session.cancel_current()
        processes = self._processes(control.process)
        for process in processes:
            try:
                process.terminate()
            except psutil.Error:
                pass
        if processes:
            _, alive = psutil.wait_procs(processes, timeout=3)
            for process in alive:
                try:
                    process.kill()
                except psutil.Error:
                    pass
        self.update(job, status="canceled", stage="任务已取消", log="任务已取消，可安全删除")
        return job

    def delete(self, job_id: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status in {"queued", "running", "paused"}:
                raise RuntimeError("请先取消正在运行或暂停的任务")
            control = self.controls.get(job_id)
            if control and not control.finished_event.is_set():
                raise RuntimeError("任务正在停止并清理资源，请稍候再删除")
            input_path = Path(job.options.input_path).resolve()
            other_inputs = {
                Path(other.options.input_path).resolve()
                for other_id, other in self.jobs.items()
                if other_id != job_id
            }
            self.jobs.pop(job_id)
            self.controls.pop(job_id, None)
        target = (JOBS_DIR / job_id).resolve()
        if target.parent != JOBS_DIR.resolve():
            raise RuntimeError("任务目录不安全，拒绝删除")
        if target.exists():
            shutil.rmtree(target)
        uploads_root = UPLOADS_DIR.resolve()
        if input_path.is_relative_to(uploads_root) and input_path not in other_inputs:
            relative = input_path.relative_to(uploads_root)
            upload_dir = uploads_root / relative.parts[0]
            if upload_dir.parent == uploads_root and upload_dir.exists():
                shutil.rmtree(upload_dir)
        return target

    def checkpoint(self, job: JobState):
        control = self._control(job.id)
        while not control.run_gate.wait(0.25):
            if control.cancel_event.is_set():
                raise JobCancelled("任务已取消")
        if control.cancel_event.is_set():
            raise JobCancelled("任务已取消")

    def persist(self, job: JobState):
        folder = JOBS_DIR / job.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "status.json").write_text(
            json.dumps(job.public(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def update(self, job: JobState, *, stage=None, progress=None, status=None, log=None):
        if stage is not None:
            job.stage = stage
        if progress is not None:
            job.progress = progress
        if status is not None:
            job.status = status
        if log:
            job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log}")
        job.updated_at = _now()
        self.persist(job)

    def _run_guarded(self, job: JobState):
        control = self._control(job.id)
        acquired = False
        try:
            while not acquired:
                self.checkpoint(job)
                acquired = GPU_LOCK.acquire(timeout=0.25)
            try:
                self.run_pipeline(job)
            except JobCancelled:
                if job.status != "canceled":
                    self.update(job, status="canceled", stage="任务已取消", log="任务已取消")
            except Exception as exc:
                control = self.controls.get(job.id)
                if control and control.cancel_event.is_set():
                    if job.status != "canceled":
                        self.update(job, status="canceled", stage="任务已取消", log="任务已取消")
                else:
                    job.error = str(exc)
                    self.update(job, status="failed", stage="处理失败", log=traceback.format_exc())
        except JobCancelled:
            if job.status != "canceled":
                self.update(job, status="canceled", stage="任务已取消", log="任务已取消")
        finally:
            session = job.cloud_session
            if session:
                try:
                    session.checkpoint = lambda: None
                    session.cleanup_job()
                except Exception:
                    pass
                finally:
                    session.close()
                    job.cloud_session = None
            cloud_audio = (JOBS_DIR / job.id / "work" / "cloud_audio.flac").resolve()
            expected_workdir = (JOBS_DIR / job.id / "work").resolve()
            preserve_staged_audio = job.status == "staged" or (
                job.status == "failed" and job.options.cloud_stage_only
            )
            if cloud_audio.parent == expected_workdir and not preserve_staged_audio:
                cloud_audio.unlink(missing_ok=True)
            if acquired:
                GPU_LOCK.release()
            control.finished_event.set()

    def run_command(
        self,
        job: JobState,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ):
        self.checkpoint(job)
        merged_env = os.environ.copy()
        merged_env["PYTHONUNBUFFERED"] = "1"
        merged_env["HF_HOME"] = str(ROOT / "hf_cache")
        if env:
            merged_env.update(env)
        process = subprocess.Popen(
            command,
            cwd=str(cwd or ROOT),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        control = self._control(job.id)
        control.process = process
        assert process.stdout is not None
        try:
            for line in process.stdout:
                line = line.strip()
                if line and not re.match(r"^\s*\d+%", line):
                    self.update(job, log=line[-600:])
            code = process.wait()
        finally:
            process.stdout.close()
            control.process = None
        self.checkpoint(job)
        if code:
            raise RuntimeError(f"命令执行失败（退出码 {code}）：{' '.join(command[:3])}")

    @staticmethod
    def media_duration(media: Path):
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(media),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())

    def run_pipeline(self, job: JobState):
        self.checkpoint(job)
        options = job.options
        media = Path(options.input_path).resolve()
        if not media.exists() or not media.is_file():
            raise FileNotFoundError(f"输入视频不存在：{media}")
        job_dir = JOBS_DIR / job.id
        workdir = job_dir / "work"
        output_dir = (
            Path(options.output_dir).expanduser().resolve()
            if options.output_dir
            else job_dir / "output"
        )
        workdir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_stem = options.output_name.strip() or media.stem
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", raw_stem).strip(" .") or "subtitle_output"
        profile = PROFILE_SETTINGS[options.profile]
        language = language_info(options.source_language)
        python = sys.executable
        duration = self.media_duration(media)
        worker_settings = job.cloud_worker_settings
        use_cloud_worker = bool(
            worker_settings and worker_settings.enabled and options.asr.kind == "local_whisper"
        )
        analysis_media = media
        if use_cloud_worker:
            self.update(
                job,
                status="running",
                stage="本地提取云识别音轨",
                progress=0.01,
                log="原视频保留在本机，仅向云节点上传单声道 FLAC 音轨",
            )
            analysis_media = workdir / "cloud_audio.flac"
            if analysis_media.exists() and analysis_media.stat().st_size:
                self.update(job, log="复用预上传阶段生成的本地 16 kHz 单声道 FLAC 音轨")
            else:
                self.run_command(
                    job,
                    [
                        "ffmpeg", "-y", "-i", str(media), "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "flac", "-compression_level", "8", str(analysis_media),
                    ],
                )
        if options.cloud_stage_only:
            if not use_cloud_worker:
                raise RuntimeError("无卡预上传必须先启用云 GPU 运算单元和本地 Whisper")
            self.update(job, stage="连接无卡模式并校验上传", progress=0.08)
            worker = CloudWhisperWorker(
                worker_settings,
                logger=lambda message: self.update(job, log=message[-600:]),
                checkpoint=lambda: self.checkpoint(job),
            )
            job.cloud_session = worker
            result = worker.stage_job_audio(job.id, analysis_media)
            worker.close()
            job.cloud_session = None
            self.update(
                job,
                status="staged",
                stage="音轨已校验，等待 GPU 开机",
                progress=0.20,
                log=(
                    f"预上传完成：{int(result['size']) / 1024 / 1024:.1f} MB，"
                    f"SHA-256 {str(result['sha256'])[:12]}…；正式处理前会再次校验"
                ),
            )
            return
        self.update(job, status="running", stage="声音活动检测", progress=0.03, log=f"视频时长 {duration:.2f} 秒")

        vad_path = workdir / "vad_segments.json"
        self.run_command(
            job,
            [python, str(ROOT / "vad_scan.py"), str(analysis_media), "--output", str(vad_path), "--mode", "3"],
        )
        vad_segments = json.loads(vad_path.read_text(encoding="utf-8"))

        worker = None
        if use_cloud_worker:
            self.update(job, stage="连接云 GPU 运算单元", progress=0.08)
            worker = CloudWhisperWorker(
                worker_settings,
                logger=lambda message: self.update(job, log=message[-600:]),
                checkpoint=lambda: self.checkpoint(job),
            )
            job.cloud_session = worker
            worker.prepare_job(job.id, analysis_media)

        events_path = workdir / "event_segments.json"
        self.update(job, stage="喘息与语音分类", progress=0.12)
        if worker:
            worker.run_event_gate(vad_path, events_path)
        else:
            self.run_command(
                job,
                [
                    python, str(ROOT / "audio_event_gate.py"), str(analysis_media),
                    "--vad", str(vad_path), "--output", str(events_path),
                ],
            )
        events = json.loads(events_path.read_text(encoding="utf-8"))

        def execute_whisper_asr(
            event_file: Path,
            target_workdir: Path,
            *,
            label: str,
            speech_threshold: float,
            nonlexical_factor: float,
        ):
            if worker:
                worker.run_asr(
                    event_file,
                    target_workdir,
                    label=label,
                    model=options.asr.model,
                    language=options.source_language,
                    speech_threshold=speech_threshold,
                    nonlexical_factor=nonlexical_factor,
                )
            else:
                self.run_command(
                    job,
                    [
                        python, str(ROOT / "asr_stage.py"), str(analysis_media),
                        "--events", str(event_file), "--workdir", str(target_workdir),
                        "--model", options.asr.model, "--language", options.source_language,
                        "--speech-threshold", str(speech_threshold),
                        "--nonlexical-factor", str(nonlexical_factor),
                    ],
                )

        def execute_whisper_review(source_file: Path, target_workdir: Path, *, label: str):
            if worker:
                worker.run_review(
                    source_file,
                    target_workdir,
                    label=label,
                    model=options.verifier_model,
                    language=options.source_language,
                )
            else:
                self.run_command(
                    job,
                    [
                        python, str(ROOT / "large_review.py"), str(analysis_media),
                        "--medium", str(source_file), "--workdir", str(target_workdir),
                        "--model", options.verifier_model, "--language", options.source_language,
                    ],
                )

        self.update(job, stage=f"{language['name']}语音识别", progress=0.25)
        if options.asr.kind == "local_whisper":
            execute_whisper_asr(
                events_path,
                workdir,
                label="primary",
                speech_threshold=profile["speech_threshold"],
                nonlexical_factor=profile["nonlexical_factor"],
            )
        elif options.asr.kind == "openai_compatible":
            def remote_progress(current, total, text):
                self.checkpoint(job)
                self.update(
                    job, progress=0.25 + 0.20 * current / max(1, total), log=text
                )

            run_remote_asr(
                analysis_media,
                events,
                options.asr,
                workdir,
                profile["speech_threshold"],
                profile["nonlexical_factor"],
                options.source_language,
                remote_progress,
            )
        else:
            raise ValueError(f"不支持的 ASR 提供方：{options.asr.kind}")

        source_path = workdir / "source_sentences.json"
        initial_source = json.loads(source_path.read_text(encoding="utf-8"))
        if initial_source and options.verifier_model and options.verifier_model != options.asr.model:
            self.update(job, stage="第二模型复核", progress=0.46)
            execute_whisper_review(source_path, workdir, label="primary")
            primary_path = workdir / "source_final.json"
        else:
            primary_path = source_path
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
        comparison_path = workdir / "model_comparison.json"
        if comparison_path.exists():
            primary = attach_asr_reviews(
                primary,
                json.loads(comparison_path.read_text(encoding="utf-8")),
            )

        recovered_count = 0
        music_recovered_count = 0
        vad_fallback_count = 0
        initial_gaps = save_gap_audit(workdir / "gaps_before_recovery.json", primary, duration)
        if (
            initial_gaps
            and options.asr.kind == "local_whisper"
            and options.verifier_model
        ):
            recovery_events = filter_events_for_gaps(
                events,
                initial_gaps,
                profile["recovery_threshold"],
                profile["nonlexical_factor"],
            )
            if profile.get("vad_gap_fallback"):
                fallback_events = vad_fallback_events_for_gaps(
                    vad_segments, initial_gaps, recovery_events
                )
                vad_fallback_count = len(fallback_events)
                recovery_events.extend(fallback_events)
                recovery_events.sort(key=lambda x: (x["start"], x["end"]))
            if recovery_events:
                self.update(
                    job,
                    stage="长空白二次召回",
                    progress=0.57,
                    log=(
                        f"召回窗口 {len(recovery_events)} 个，"
                        f"其中 VAD 兜底 {vad_fallback_count} 个"
                    ),
                )
                recovery_root = workdir / "recovery"
                recovery_root.mkdir(exist_ok=True)
                recovery_events_path = recovery_root / "events.json"
                recovery_events_path.write_text(
                    json.dumps(recovery_events, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                execute_whisper_asr(
                    recovery_events_path,
                    recovery_root,
                    label="gap_recovery",
                    speech_threshold=profile["recovery_threshold"],
                    nonlexical_factor=max(1.0, profile["nonlexical_factor"] - 0.1),
                )
                recovery_medium = recovery_root / "source_sentences.json"
                if json.loads(recovery_medium.read_text(encoding="utf-8")):
                    execute_whisper_review(
                        recovery_medium, recovery_root, label="gap_recovery"
                    )
                    recovery_final = json.loads(
                        (recovery_root / "source_final.json").read_text(encoding="utf-8")
                    )
                    comparisons = json.loads(
                        (recovery_root / "model_comparison.json").read_text(encoding="utf-8")
                    )
                    accepted = accepted_recovery_rows(
                        recovery_final, comparisons, profile["consensus_threshold"]
                    )
                    before_recovery = len(primary)
                    primary = merge_recovery(primary, accepted)
                    recovered_count = len(primary) - before_recovery

        if (
            options.asr.kind == "local_whisper"
            and options.verifier_model
        ):
            music_events = filter_events_for_uncovered_speech(
                events,
                primary,
                profile["recovery_threshold"],
                profile["nonlexical_factor"],
            )
            if music_events:
                self.update(
                    job,
                    stage="自动复核未覆盖弱对白",
                    progress=0.63,
                    log=f"复核未覆盖的弱对白候选 {len(music_events)} 个",
                )
                music_root = workdir / "music_recovery"
                music_root.mkdir(exist_ok=True)
                music_events_path = music_root / "events.json"
                music_events_path.write_text(
                    json.dumps(music_events, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                execute_whisper_asr(
                    music_events_path,
                    music_root,
                    label="weak_speech_recovery",
                    speech_threshold=profile["recovery_threshold"],
                    nonlexical_factor=max(1.0, profile["nonlexical_factor"] - 0.2),
                )
                music_medium = music_root / "source_sentences.json"
                if json.loads(music_medium.read_text(encoding="utf-8")):
                    execute_whisper_review(
                        music_medium, music_root, label="weak_speech_recovery"
                    )
                    music_final = json.loads(
                        (music_root / "source_final.json").read_text(encoding="utf-8")
                    )
                    music_comparisons = json.loads(
                        (music_root / "model_comparison.json").read_text(encoding="utf-8")
                    )
                    accepted_music = accepted_recovery_rows(
                        music_final,
                        music_comparisons,
                        max(0.58, profile["consensus_threshold"]),
                        "music_recovery_consensus",
                    )
                    before_music = len(primary)
                    primary = merge_recovery(primary, accepted_music)
                    music_recovered_count = len(primary) - before_music
                    self.update(
                        job,
                        log=f"未覆盖弱对白经双模型确认补回 {music_recovered_count} 条",
                    )

        if worker:
            self.update(job, stage="回收云端临时音轨", progress=0.68)
            worker.cleanup_job()
            worker.close()
            job.cloud_session = None

        self.update(job, stage="逐句翻译与否定词审计", progress=0.70)
        def translation_progress(current, total, _):
            self.checkpoint(job)
            self.update(
                job,
                progress=0.70 + 0.12 * current / max(1, total),
                log=(f"翻译 {current}/{total}" if current % 10 == 0 else None),
            )

        translated = translate_cues(
            primary,
            options.translator,
            translation_progress,
            source_language=options.source_language,
            target_language=options.target_language,
        )
        text_review_audit = {
            "enabled": False,
            "cue_count": len(translated),
            "changed_count": 0,
            "rejected_count": 0,
            "invalid_batches": 0,
            "changes": [],
            "rejected": [],
            "glossary": [],
        }
        text_review_audit_path = None
        if translated:
            self.update(job, stage="最终文本校正与全局一致性检查", progress=0.83)

            def text_review_progress(current, total, message):
                self.checkpoint(job)
                self.update(
                    job,
                    progress=0.83 + 0.05 * current / max(1, total),
                    log=message,
                )

            translated, text_review_audit = review_cues(
                translated,
                options.text_reviewer,
                text_review_progress,
                source_language=options.source_language,
            )
            text_review_audit_path = output_dir / f"{stem}_最终文本校正记录.json"
            text_review_audit_path.write_text(
                json.dumps(text_review_audit, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.update(
                job,
                log=(
                    f"最终文本校正完成：修改 {text_review_audit['changed_count']} 条，"
                    f"安全回退 {text_review_audit['rejected_count']} 条"
                ),
            )
        review_cues = finalize_cues(
            translated,
            min_duration=0.85,
            remove_periods=options.remove_chinese_periods,
            publish=False,
        )
        publish_cues = finalize_cues(
            translated,
            min_duration=0.85,
            remove_periods=options.remove_chinese_periods,
            publish=True,
        )
        review_files = write_subtitles(
            review_cues, output_dir, stem, "高置信校对版", options.source_language
        )
        publish_files = write_subtitles(
            publish_cues, output_dir, stem, "观看版", options.source_language
        )
        final_cues = publish_cues if options.publish_mode else review_cues
        final_files = publish_files if options.publish_mode else review_files

        summary = quality_summary(final_cues, duration, activity_segments=vad_segments)
        summary["profile"] = options.profile
        summary["recovered_cues"] = recovered_count
        summary["music_recovered_cues"] = music_recovered_count
        summary["vad_fallback_segments"] = vad_fallback_count
        summary["input_duration"] = duration
        summary["source_language"] = language["name"]
        warning_items = [
            {
                "start": float(row["start"]),
                "source": row.get("source", ""),
                "zh": row.get("zh", ""),
                "warnings": row.get("translation_warnings", []),
            }
            for row in translated
            if row.get("translation_warnings")
        ]
        summary["translation_warning_count"] = len(warning_items)
        summary["translation_warning_items"] = warning_items[:100]
        summary["text_review_enabled"] = bool(text_review_audit["enabled"])
        summary["text_review_model"] = text_review_audit.get("model", "")
        summary["text_review_changed_count"] = text_review_audit["changed_count"]
        summary["text_review_rejected_count"] = text_review_audit["rejected_count"]
        summary["text_review_invalid_batches"] = text_review_audit["invalid_batches"]
        report_path = output_dir / f"{stem}_自动质量报告.md"
        report_path.write_text(self.quality_report(summary), encoding="utf-8")

        if options.create_soft_subtitle_video and final_cues:
            self.update(job, stage="封装软字幕视频", progress=0.90)
            video_out = output_dir / f"{stem}_中文字幕软字幕.mp4"
            mux_soft_subtitles(
                media,
                final_files["cn_srt"],
                video_out,
                "简体中文",
                run=lambda command, cwd=None: self.run_command(job, command, cwd=cwd),
            )
            job.outputs["soft_video"] = str(video_out)
        elif options.create_soft_subtitle_video:
            self.update(job, log="未检测到可靠对白，跳过软字幕视频封装")

        if options.create_hard_subtitle_video and final_cues:
            self.update(job, stage="压制硬字幕视频", progress=0.94)
            hard_video_out = output_dir / f"{stem}_中文字幕硬字幕.mp4"
            mux_hard_subtitles(
                media,
                final_files["cn_srt"],
                hard_video_out,
                run=lambda command, cwd=None: self.run_command(job, command, cwd=cwd),
            )
            job.outputs["hard_video"] = str(hard_video_out)
        elif options.create_hard_subtitle_video:
            self.update(job, log="未检测到可靠对白，跳过硬字幕视频压制")

        for prefix, files in (("review", review_files), ("publish", publish_files)):
            for name, path in files.items():
                job.outputs[f"{prefix}_{name}"] = str(path)
        job.outputs["quality_report"] = str(report_path)
        if text_review_audit_path:
            job.outputs["text_review_audit"] = str(text_review_audit_path)
        self.update(
            job,
            status="completed",
            stage="处理完成",
            progress=1.0,
            log=f"自动质量复核补回长空白 {recovered_count} 条、短弱对白 {music_recovered_count} 条",
        )

    @staticmethod
    def quality_report(summary: dict[str, Any]):
        text_review_status = "已启用" if summary["text_review_enabled"] else "未启用"
        if summary["text_review_model"]:
            text_review_status += f"（{summary['text_review_model']}）"
        gaps = "\n".join(
            f"- {x['start']:.2f} ～ {x['end']:.2f}（{x['duration']:.2f} 秒，"
            f"其中 VAD 活动 {x.get('activity_seconds', 0):.2f} 秒）"
            for x in summary["long_gaps"]
        ) or "- 无超过 30 秒的空白区间"
        warnings = "\n".join(
            f"- {x['start']:.2f} 秒｜原文：{x['source']}｜译文：{x['zh']}｜"
            f"原因：{'；'.join(x['warnings'])}"
            for x in summary.get("translation_warning_items", [])
        ) or "- 无"
        return f"""# 自动字幕质量报告

- 源语言：{summary['source_language']}
- 策略：{summary['profile']}
- 字幕条数：{summary['cue_count']}
- 字幕显示总时长：{summary['display_seconds']} 秒
- 小于 0.85 秒：{summary['under_085_seconds']} 条
- 恰好 2 秒：{summary['exact_two_seconds']} 条
- 时间轴重叠：{summary['overlaps']} 处
- 未确认占位符：{summary['placeholders']} 处
- 翻译审计警告：{summary['translation_warning_count']} 条（校对版以“【需校对】”标出）
- 最终文本校正：{text_review_status}
- 校正实际修改：{summary['text_review_changed_count']} 条
- 校正安全回退：{summary['text_review_rejected_count']} 条（结构无效批次 {summary['text_review_invalid_batches']} 个）
- 中文字幕句号：{summary['chinese_periods']} 个
- 长空白二次召回：补回 {summary['recovered_cues']} 条
- 自动短弱对白复核：补回 {summary.get('music_recovered_cues', 0)} 条
- VAD 长空白兜底：复查 {summary['vad_fallback_segments']} 段
- VAD 活动覆盖率：{summary['activity_coverage_percent']}%
- 最长空白：{summary['longest_gap']} 秒

## 超过 30 秒的空白区间

{gaps}

## 需要人工校对的译文

{warnings}

## 解释

长空白不等于漏识别：可能是静音、喘息、呻吟、水声或没有语言的信息。本报告同时显示空白区里的 VAD 活动时长，便于判断漏识别风险。常规补回经过声音事件门控；VAD 兜底仅在长空白区复查门控漏掉的活动片段，两者都必须通过双模型一致性检查后才能进入字幕。
"""


manager = JobManager()
