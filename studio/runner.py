from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .quality import PROFILE_SETTINGS, finalize_cues, quality_summary
from .recall import (
    accepted_recovery_rows,
    filter_events_for_gaps,
    merge_recovery,
    save_gap_audit,
)
from .remote_asr import run_remote_asr
from .schemas import JobOptions
from .subtitles import mux_hard_subtitles, mux_soft_subtitles, write_subtitles
from .translation import translate_cues


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "studio_data"
JOBS_DIR = DATA_DIR / "jobs"
UPLOADS_DIR = DATA_DIR / "uploads"
GPU_LOCK = threading.Lock()


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

    def public(self):
        safe_options = self.options.model_dump()
        safe_options["asr"]["api_key"] = ""
        safe_options["translator"]["api_key"] = ""
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
                if status in {"queued", "running"}:
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

    def create(self, options: JobOptions):
        job = JobState(id=uuid.uuid4().hex[:12], options=options)
        with self.lock:
            self.jobs[job.id] = job
        self.persist(job)
        threading.Thread(target=self._run_guarded, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str):
        return self.jobs.get(job_id)

    def list(self):
        return sorted((x.public() for x in self.jobs.values()), key=lambda x: x["created_at"], reverse=True)

    def archive(self, job_id: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status in {"queued", "running"}:
                raise RuntimeError("运行中的任务不能移除")
            self.jobs.pop(job_id)
        source = JOBS_DIR / job_id
        archive_dir = DATA_DIR / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / f"{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if source.exists():
            source.replace(target)
        return target

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
        with GPU_LOCK:
            try:
                self.run_pipeline(job)
            except Exception as exc:
                job.error = str(exc)
                self.update(job, status="failed", stage="处理失败", log=traceback.format_exc())

    def run_command(self, job: JobState, command: list[str], env: dict[str, str] | None = None):
        merged_env = os.environ.copy()
        merged_env["PYTHONUNBUFFERED"] = "1"
        merged_env["HF_HOME"] = str(ROOT / "hf_cache")
        if env:
            merged_env.update(env)
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if line and not re.match(r"^\s*\d+%", line):
                self.update(job, log=line[-600:])
        code = process.wait()
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
        options = job.options
        media = Path(options.input_path).resolve()
        if not media.exists() or not media.is_file():
            raise FileNotFoundError(f"输入视频不存在：{media}")
        job_dir = JOBS_DIR / job.id
        workdir = job_dir / "work"
        output_dir = job_dir / "output"
        workdir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_stem = options.output_name.strip() or media.stem
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", raw_stem).strip(" .") or "subtitle_output"
        profile = PROFILE_SETTINGS[options.profile]
        python = sys.executable
        duration = self.media_duration(media)
        self.update(job, status="running", stage="声音活动检测", progress=0.03, log=f"视频时长 {duration:.2f} 秒")

        vad_path = workdir / "vad_segments.json"
        self.run_command(
            job,
            [python, str(ROOT / "vad_scan.py"), str(media), "--output", str(vad_path), "--mode", "3"],
        )

        events_path = workdir / "event_segments.json"
        self.update(job, stage="喘息与语音分类", progress=0.12)
        self.run_command(
            job,
            [
                python, str(ROOT / "audio_event_gate.py"), str(media), "--vad", str(vad_path),
                "--output", str(events_path),
            ],
        )
        events = json.loads(events_path.read_text(encoding="utf-8"))

        self.update(job, stage="日语语音识别", progress=0.25)
        if options.asr.kind == "local_whisper":
            self.run_command(
                job,
                [
                    python, str(ROOT / "asr_stage.py"), str(media), "--events", str(events_path),
                    "--workdir", str(workdir), "--model", options.asr.model,
                    "--speech-threshold", str(profile["speech_threshold"]),
                    "--nonlexical-factor", str(profile["nonlexical_factor"]),
                ],
            )
        elif options.asr.kind == "openai_compatible":
            run_remote_asr(
                media,
                events,
                options.asr,
                workdir,
                profile["speech_threshold"],
                profile["nonlexical_factor"],
                options.source_language,
                lambda current, total, text: self.update(
                    job, progress=0.25 + 0.20 * current / max(1, total), log=text
                ),
            )
        else:
            raise ValueError(f"不支持的 ASR 提供方：{options.asr.kind}")

        ja_source = workdir / "ja_sentences.json"
        initial_ja = json.loads(ja_source.read_text(encoding="utf-8"))
        if initial_ja and options.verifier_model and options.verifier_model != options.asr.model:
            self.update(job, stage="第二模型复核", progress=0.46)
            self.run_command(
                job,
                [
                    python, str(ROOT / "large_review.py"), str(media), "--medium", str(ja_source),
                    "--workdir", str(workdir), "--model", options.verifier_model,
                ],
            )
            primary_path = workdir / "ja_final.json"
        else:
            primary_path = ja_source
        primary = json.loads(primary_path.read_text(encoding="utf-8"))

        recovered_count = 0
        initial_gaps = save_gap_audit(workdir / "gaps_before_recovery.json", primary, duration)
        if (
            options.enable_gap_recovery
            and initial_gaps
            and options.asr.kind == "local_whisper"
            and options.verifier_model
        ):
            recovery_events = filter_events_for_gaps(
                events,
                initial_gaps,
                profile["recovery_threshold"],
                profile["nonlexical_factor"],
            )
            if recovery_events:
                self.update(job, stage="长空白二次召回", progress=0.57, log=f"临界声音窗口 {len(recovery_events)} 个")
                recovery_root = workdir / "recovery"
                recovery_root.mkdir(exist_ok=True)
                recovery_events_path = recovery_root / "events.json"
                recovery_events_path.write_text(
                    json.dumps(recovery_events, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self.run_command(
                    job,
                    [
                        python, str(ROOT / "asr_stage.py"), str(media), "--events", str(recovery_events_path),
                        "--workdir", str(recovery_root), "--model", options.asr.model,
                        "--speech-threshold", str(profile["recovery_threshold"]),
                        "--nonlexical-factor", str(max(1.0, profile["nonlexical_factor"] - 0.1)),
                    ],
                )
                recovery_medium = recovery_root / "ja_sentences.json"
                if json.loads(recovery_medium.read_text(encoding="utf-8")):
                    self.run_command(
                        job,
                        [
                            python, str(ROOT / "large_review.py"), str(media), "--medium", str(recovery_medium),
                            "--workdir", str(recovery_root), "--model", options.verifier_model,
                        ],
                    )
                    recovery_final = json.loads((recovery_root / "ja_final.json").read_text(encoding="utf-8"))
                    comparisons = json.loads(
                        (recovery_root / "model_comparison.json").read_text(encoding="utf-8")
                    )
                    accepted = accepted_recovery_rows(
                        recovery_final, comparisons, profile["consensus_threshold"]
                    )
                    recovered_count = len(accepted)
                    primary = merge_recovery(primary, accepted)

        self.update(job, stage="逐句翻译与否定词审计", progress=0.70)
        translated = translate_cues(
            primary,
            options.translator,
            lambda current, total, _: self.update(
                job, progress=0.70 + 0.16 * current / max(1, total), log=(f"翻译 {current}/{total}" if current % 10 == 0 else None)
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
        review_files = write_subtitles(review_cues, output_dir, stem, "高置信校对版")
        publish_files = write_subtitles(publish_cues, output_dir, stem, "观看版")
        final_cues = publish_cues if options.publish_mode else review_cues
        final_files = publish_files if options.publish_mode else review_files

        summary = quality_summary(final_cues, duration)
        summary["profile"] = options.profile
        summary["recovered_cues"] = recovered_count
        summary["input_duration"] = duration
        report_path = output_dir / f"{stem}_自动质量报告.md"
        report_path.write_text(self.quality_report(summary), encoding="utf-8")

        if options.create_soft_subtitle_video and final_cues:
            self.update(job, stage="封装软字幕视频", progress=0.90)
            video_out = output_dir / f"{stem}_中文字幕软字幕.mp4"
            mux_soft_subtitles(media, final_files["cn_srt"], video_out, "简体中文")
            job.outputs["soft_video"] = str(video_out)
        elif options.create_soft_subtitle_video:
            self.update(job, log="未检测到可靠对白，跳过软字幕视频封装")

        if options.create_hard_subtitle_video and final_cues:
            self.update(job, stage="压制硬字幕视频", progress=0.94)
            hard_video_out = output_dir / f"{stem}_中文字幕硬字幕.mp4"
            mux_hard_subtitles(media, final_files["cn_srt"], hard_video_out)
            job.outputs["hard_video"] = str(hard_video_out)
        elif options.create_hard_subtitle_video:
            self.update(job, log="未检测到可靠对白，跳过硬字幕视频压制")

        for prefix, files in (("review", review_files), ("publish", publish_files)):
            for name, path in files.items():
                job.outputs[f"{prefix}_{name}"] = str(path)
        job.outputs["quality_report"] = str(report_path)
        self.update(job, status="completed", stage="处理完成", progress=1.0, log=f"二次召回补回 {recovered_count} 条")

    @staticmethod
    def quality_report(summary: dict[str, Any]):
        gaps = "\n".join(
            f"- {x['start']:.2f} ～ {x['end']:.2f}（{x['duration']:.2f} 秒）"
            for x in summary["long_gaps"]
        ) or "- 无超过 30 秒的空白区间"
        return f"""# 自动字幕质量报告

- 策略：{summary['profile']}
- 字幕条数：{summary['cue_count']}
- 字幕显示总时长：{summary['display_seconds']} 秒
- 小于 0.85 秒：{summary['under_085_seconds']} 条
- 恰好 2 秒：{summary['exact_two_seconds']} 条
- 时间轴重叠：{summary['overlaps']} 处
- 未确认占位符：{summary['placeholders']} 处
- 中文字幕句号：{summary['chinese_periods']} 个
- 长空白二次召回：补回 {summary['recovered_cues']} 条
- 最长空白：{summary['longest_gap']} 秒

## 超过 30 秒的空白区间

{gaps}

## 解释

长空白不等于漏识别：可能是静音、喘息、呻吟、水声或没有语言的信息。本报告列出它们供复核；自动补回只接受声音事件门控和双模型一致性同时通过的片段。
"""


manager = JobManager()
