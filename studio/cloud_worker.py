from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex
import time
from pathlib import Path
from typing import Callable

from .config import ROOT
from .schemas import CloudWorkerSettings


class CloudWorkerError(RuntimeError):
    pass


def _paramiko():
    try:
        import paramiko
    except ImportError as exc:
        raise CloudWorkerError(
            "缺少本地 SSH 组件 paramiko，请重新运行依赖安装后再连接云节点"
        ) from exc
    return paramiko


def _validated(settings: CloudWorkerSettings) -> CloudWorkerSettings:
    if not settings.host.strip():
        raise CloudWorkerError("请填写云服务器地址")
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", settings.host.strip()):
        raise CloudWorkerError("云服务器地址包含不支持的字符")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", settings.username.strip()):
        raise CloudWorkerError("云服务器用户名包含不支持的字符")
    if not settings.password and not settings.private_key_path:
        raise CloudWorkerError("请填写 SSH 密码或私钥路径")
    if settings.private_key_path and not Path(settings.private_key_path).expanduser().is_file():
        raise CloudWorkerError("SSH 私钥文件不存在")
    remote_dir = settings.remote_dir.strip().rstrip("/")
    if not remote_dir.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9._/-]+", remote_dir):
        raise CloudWorkerError("云端工作目录必须是只含英文、数字、点、横线的绝对路径")
    result = settings.model_copy(deep=True)
    result.host = result.host.strip()
    result.username = result.username.strip()
    result.remote_dir = remote_dir
    model_dir = settings.model_dir.strip().rstrip("/")
    if not model_dir.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9._/-]+", model_dir):
        raise CloudWorkerError("云端模型目录必须是只含英文、数字、点、横线的绝对路径")
    result.model_dir = model_dir
    return result


class CloudWhisperWorker:
    def __init__(
        self,
        settings: CloudWorkerSettings,
        *,
        logger: Callable[[str], None] | None = None,
        checkpoint: Callable[[], None] | None = None,
    ):
        self.settings = _validated(settings)
        self.logger = logger or (lambda _message: None)
        self.checkpoint = checkpoint or (lambda: None)
        self.client = None
        self.sftp = None
        self.remote_job_dir = ""
        self.active_control_file = ""

    def connect(self):
        paramiko = _paramiko()
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = {
            "hostname": self.settings.host,
            "port": self.settings.port,
            "username": self.settings.username,
            "password": self.settings.password or None,
            "key_filename": str(Path(self.settings.private_key_path).expanduser())
            if self.settings.private_key_path
            else None,
            "timeout": 15,
            "banner_timeout": 20,
            "auth_timeout": 20,
            "look_for_keys": not bool(self.settings.password or self.settings.private_key_path),
            "allow_agent": not bool(self.settings.password or self.settings.private_key_path),
        }
        client.connect(**kwargs)
        self.client = client
        self.sftp = client.open_sftp()
        return self

    def close(self):
        if self.sftp:
            try:
                self.sftp.close()
            except (EOFError, OSError):
                pass
            finally:
                self.sftp = None
        if self.client:
            try:
                self.client.close()
            finally:
                self.client = None

    def _exec(
        self,
        command: str,
        timeout: float | None = None,
        *,
        controllable: bool = False,
    ) -> str:
        if not self.client:
            raise CloudWorkerError("云节点尚未连接")
        channel = self.client.get_transport().open_session()
        if controllable:
            self.active_control_file = posixpath.join(self.remote_job_dir, ".active-process")
            control = shlex.quote(self.active_control_file)
            inner = (
                f"echo $$ > {control}; {command}; status=$?; "
                f"rm -f {control}; exit $status"
            )
            # util-linux setsid may fork when its caller is already a process-group
            # leader. Without --wait the SSH channel can report success while the
            # actual GPU command is still starting, so callers race to download
            # output files that do not exist yet.
            command = "setsid --wait bash -lc " + shlex.quote(inner)
        channel.exec_command(command)
        output: list[str] = []
        errors: list[str] = []
        started = time.monotonic()
        try:
            while not channel.exit_status_ready():
                self.checkpoint()
                if channel.recv_ready():
                    text = channel.recv(65536).decode("utf-8", errors="replace")
                    output.append(text)
                    for line in text.splitlines():
                        if line.strip():
                            self.logger(line.strip())
                if channel.recv_stderr_ready():
                    text = channel.recv_stderr(65536).decode("utf-8", errors="replace")
                    errors.append(text)
                    for line in text.splitlines():
                        if line.strip():
                            self.logger(line.strip())
                if timeout and time.monotonic() - started > timeout:
                    raise CloudWorkerError("云节点命令执行超时")
                time.sleep(0.15)
            while channel.recv_ready():
                output.append(channel.recv(65536).decode("utf-8", errors="replace"))
            while channel.recv_stderr_ready():
                errors.append(channel.recv_stderr(65536).decode("utf-8", errors="replace"))
            status = channel.recv_exit_status()
            combined = "".join(output)
            error_text = "".join(errors)
            if status:
                raise CloudWorkerError(
                    f"云节点命令失败（退出码 {status}）：{error_text.strip() or combined.strip()}"
                )
            return combined
        finally:
            channel.close()
            if controllable:
                self.active_control_file = ""

    def _signal_current(self, signal: str):
        if not self.client or not self.active_control_file:
            return
        control = shlex.quote(self.active_control_file)
        command = (
            f"if [ -f {control} ]; then kill -{signal} -- -$(cat {control}) 2>/dev/null || true; fi"
        )
        channel = self.client.get_transport().open_session()
        try:
            channel.exec_command("bash -lc " + shlex.quote(command))
            channel.recv_exit_status()
        finally:
            channel.close()

    def pause_current(self):
        self._signal_current("STOP")

    def resume_current(self):
        self._signal_current("CONT")

    def cancel_current(self):
        self._signal_current("TERM")

    def test_connection(self) -> dict[str, str]:
        self.connect()
        try:
            output = self._exec(
                "printf 'system='; uname -srm; "
                "printf 'gpu='; (nvidia-smi --query-gpu=name,memory.total "
                "--format=csv,noheader 2>/dev/null || printf 'not-found')",
                timeout=20,
            )
            values = {}
            for line in output.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip()
            return values
        finally:
            self.close()

    def bootstrap(self) -> dict[str, str]:
        if not self.client:
            self.connect()
        remote = shlex.quote(self.settings.remote_dir)
        script = f"""set -e
export OMP_NUM_THREADS=4
if ! command -v python3 >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ]; then apt-get update && apt-get install -y python3 python3-venv; else sudo -n apt-get update && sudo -n apt-get install -y python3 python3-venv; fi
fi
if ! python3 -m venv --help >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ]; then apt-get update && apt-get install -y python3-venv; else sudo -n apt-get update && sudo -n apt-get install -y python3-venv; fi
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ]; then apt-get update && apt-get install -y ffmpeg; else sudo -n apt-get update && sudo -n apt-get install -y ffmpeg; fi
fi
mkdir -p {remote}/studio
if [ ! -f {remote}/.worker-ready-v2 ]; then
  if [ ! -x {remote}/.venv/bin/python ]; then python3 -m venv --system-site-packages {remote}/.venv; fi
  {remote}/.venv/bin/python -m pip install --upgrade pip
  {remote}/.venv/bin/python -m pip install numpy openai-whisper transformers
fi
{remote}/.venv/bin/python -c 'import torch, whisper; print("torch=" + torch.__version__); print("cuda=" + str(torch.cuda.is_available())); assert torch.cuda.is_available(), "云节点 PyTorch 未启用 CUDA，请更换带 CUDA/PyTorch 的 GPU 镜像"'
touch {remote}/.worker-ready-v2
"""
        output = self._exec("bash -lc " + shlex.quote(script), timeout=1800)
        return {"output": output.strip(), "remote_dir": self.settings.remote_dir}

    def bootstrap_accuracy(self) -> dict[str, str]:
        if not self.client:
            self.connect()
        remote = shlex.quote(self.settings.remote_dir)
        models = shlex.quote(self.settings.model_dir)
        script = f"""set -euo pipefail
export OMP_NUM_THREADS=4
mkdir -p {models}
if [ -f {models}/.accuracy-ready-v1 ] \
  && [ -f {models}/weights/Qwen3-ASR-1.7B/config.json ] \
  && [ -f {models}/weights/Qwen3-ForcedAligner-0.6B/config.json ] \
  && [ -f {models}/weights/cohere-transcribe-03-2026/config.json ] \
  && [ -f {models}/weights/whisper/large-v3.pt ]; then
  echo "Accuracy ensemble already installed in {models}"
  exit 0
fi
available_kb=$(df -Pk {models} | awk 'NR==2 {{print $4}}')
if [ -z "$available_kb" ] || [ "$available_kb" -lt 36700160 ]; then
  echo "Accuracy models require at least 35 GB free in {models}; available_kb=${{available_kb:-unknown}}" >&2
  exit 23
fi
mkdir -p {models}/envs {models}/weights {models}/manifests {models}/tmp
export TMPDIR={models}/tmp
if [ ! -x {models}/envs/qwen/bin/python ]; then
  python3 -m venv --system-site-packages {models}/envs/qwen
fi
if [ ! -x {models}/envs/cohere/bin/python ]; then
  python3 -m venv --system-site-packages {models}/envs/cohere
fi
(
  {models}/envs/qwen/bin/python -m pip install -U pip
  {models}/envs/qwen/bin/python -m pip install -U qwen-asr modelscope soundfile nagisa soynlp
) & qwen_pip=$!
(
  {models}/envs/cohere/bin/python -m pip install -U pip
  {models}/envs/cohere/bin/python -m pip install -U 'transformers>=5.4.0' accelerate sentencepiece protobuf soundfile librosa modelscope
) & cohere_pip=$!
wait "$qwen_pip"
wait "$cohere_pip"
(
  [ -f {models}/weights/Qwen3-ASR-1.7B/config.json ] || {models}/envs/qwen/bin/modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir {models}/weights/Qwen3-ASR-1.7B
) & qwen_model=$!
(
  [ -f {models}/weights/Qwen3-ForcedAligner-0.6B/config.json ] || {models}/envs/qwen/bin/modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir {models}/weights/Qwen3-ForcedAligner-0.6B
) & align_model=$!
(
  [ -f {models}/weights/cohere-transcribe-03-2026/config.json ] || {models}/envs/cohere/bin/modelscope download --model CohereLabs/cohere-transcribe-03-2026 --local_dir {models}/weights/cohere-transcribe-03-2026
) & cohere_model=$!
wait "$qwen_model"
wait "$align_model"
wait "$cohere_model"
mkdir -p {models}/weights/whisper
if [ ! -f {models}/weights/whisper/large-v3.pt ]; then
  if [ -f /root/.cache/whisper/large-v3.pt ]; then
    cp /root/.cache/whisper/large-v3.pt {models}/weights/whisper/large-v3.pt
  else
    {remote}/.venv/bin/python -c "import whisper; whisper.load_model('large-v3', device='cpu', download_root='{self.settings.model_dir}/weights/whisper')"
  fi
fi
for name in Qwen3-ASR-1.7B Qwen3-ForcedAligner-0.6B cohere-transcribe-03-2026; do
  (cd {models}/weights/$name && find . -type f -not -path './.git/*' -print0 | sort -z | xargs -0 sha256sum > {models}/manifests/$name.sha256)
done
sha256sum {models}/weights/whisper/large-v3.pt > {models}/manifests/whisper-large-v3.sha256
{models}/envs/qwen/bin/python -c 'import torch, qwen_asr; assert torch.cuda.is_available()'
{models}/envs/cohere/bin/python -c 'import torch, transformers; assert torch.cuda.is_available(); print(transformers.__version__)'
touch {models}/.accuracy-ready-v1
df -h {models}
"""
        output = self._exec("bash -lc " + shlex.quote(script), timeout=7200)
        return {
            "output": output.strip(),
            "remote_dir": self.settings.remote_dir,
            "model_dir": self.settings.model_dir,
        }

    def _mkdir(self, remote_path: str):
        self._exec("mkdir -p " + shlex.quote(remote_path), timeout=30)

    def _upload(self, local_path: Path, remote_path: str, label: str):
        if not self.sftp:
            raise CloudWorkerError("云节点文件通道尚未连接")
        size = max(1, local_path.stat().st_size)
        last_percent = -1

        def callback(sent: int, _total: int):
            nonlocal last_percent
            self.checkpoint()
            percent = int(sent * 100 / size)
            if percent >= last_percent + 10 or percent == 100:
                last_percent = percent
                self.logger(f"{label} {percent}%")

        self.sftp.put(str(local_path), remote_path, callback=callback, confirm=True)

    def _local_file_info(self, local_path: Path) -> tuple[int, str]:
        size = local_path.stat().st_size
        digest = hashlib.sha256()
        with local_path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                self.checkpoint()
                digest.update(chunk)
        return size, digest.hexdigest()

    def _local_prefix_sha256(self, local_path: Path, length: int) -> str:
        remaining = length
        digest = hashlib.sha256()
        with local_path.open("rb") as stream:
            while remaining:
                self.checkpoint()
                chunk = stream.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise CloudWorkerError("本地音轨长度在上传期间发生变化")
                digest.update(chunk)
                remaining -= len(chunk)
        return digest.hexdigest()

    def _remote_file_info(self, remote_path: str) -> tuple[int, str] | None:
        quoted = shlex.quote(remote_path)
        output = self._exec(
            f"if [ -f {quoted} ]; then stat -c %s {quoted}; sha256sum {quoted} | cut -d' ' -f1; fi",
            timeout=900,
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return None
        if len(lines) != 2 or not lines[0].isdigit() or not re.fullmatch(r"[0-9a-f]{64}", lines[1]):
            raise CloudWorkerError("云端文件校验结果格式异常")
        return int(lines[0]), lines[1]

    def _upload_resumable(self, local_path: Path, remote_path: str, label: str):
        if not self.sftp:
            raise CloudWorkerError("云节点文件通道尚未连接")
        size = local_path.stat().st_size
        try:
            remote_size = self.sftp.stat(remote_path).st_size
        except FileNotFoundError:
            remote_size = 0
        if remote_size > size:
            self._exec("rm -f -- " + shlex.quote(remote_path), timeout=30)
            remote_size = 0
        elif remote_size:
            remote_info = self._remote_file_info(remote_path)
            local_prefix = self._local_prefix_sha256(local_path, remote_size)
            if remote_info != (remote_size, local_prefix):
                self.logger("云端临时分片前缀校验失败，丢弃后从头重传")
                self._exec("rm -f -- " + shlex.quote(remote_path), timeout=30)
                remote_size = 0
            elif remote_size < size:
                self.logger(f"检测到完整临时分片，从 {remote_size / 1024 / 1024:.1f} MB 处断点续传")

        sent = remote_size
        last_percent = int(sent * 100 / max(1, size)) - 10
        with local_path.open("rb") as source:
            source.seek(remote_size)
            with self.sftp.file(remote_path, "ab" if remote_size else "wb") as target:
                target.set_pipelined(True)
                while chunk := source.read(1024 * 1024):
                    self.checkpoint()
                    target.write(chunk)
                    sent += len(chunk)
                    percent = int(sent * 100 / max(1, size))
                    if percent >= last_percent + 10 or percent == 100:
                        last_percent = percent
                        self.logger(f"{label} {percent}%")

    def _reconnect(self):
        self.close()
        time.sleep(1)
        self.connect()

    def _ensure_verified_audio(self, audio_path: Path) -> dict[str, object]:
        if not self.remote_job_dir:
            raise CloudWorkerError("尚未设置云端任务目录")
        remote_audio = posixpath.join(self.remote_job_dir, "audio.flac")
        remote_part = posixpath.join(self.remote_job_dir, "audio.flac.uploading")
        remote_manifest = posixpath.join(self.remote_job_dir, "audio.ready.json")
        local_size, local_sha256 = self._local_file_info(audio_path)
        expected = (local_size, local_sha256)
        if self._remote_file_info(remote_audio) == expected:
            self.logger(f"复用已校验云端音轨 · {local_size / 1024 / 1024:.1f} MB · SHA-256 {local_sha256[:12]}…")
            return {"size": local_size, "sha256": local_sha256, "reused": True}

        for attempt in range(1, 4):
            self.checkpoint()
            try:
                self._upload_resumable(
                    audio_path,
                    remote_part,
                    f"上传音轨（连接尝试 {attempt}/3）",
                )
                remote_info = self._remote_file_info(remote_part)
            except (EOFError, OSError, _paramiko().SSHException) as exc:
                if attempt == 3:
                    raise CloudWorkerError(
                        f"音轨上传连续 3 次连接中断：{type(exc).__name__}"
                    ) from exc
                self.logger(
                    f"上传连接中断（{type(exc).__name__}），正在重连并从已校验分片继续"
                )
                self._reconnect()
                continue
            if remote_info == expected:
                manifest = json.dumps(
                    {"version": 1, "size": local_size, "sha256": local_sha256},
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                manifest_part = remote_manifest + ".uploading"
                command = (
                    f"mv -f -- {shlex.quote(remote_part)} {shlex.quote(remote_audio)}; "
                    f"printf '%s\\n' {shlex.quote(manifest)} > {shlex.quote(manifest_part)}; "
                    f"mv -f -- {shlex.quote(manifest_part)} {shlex.quote(remote_manifest)}"
                )
                self._exec(command, timeout=30)
                self.logger(f"音轨校验通过 · {local_size / 1024 / 1024:.1f} MB · SHA-256 {local_sha256[:12]}…")
                return {"size": local_size, "sha256": local_sha256, "reused": False}
            self.logger(f"音轨校验失败，第 {attempt}/3 次传输不完整，准备校验分片并续传")

        self._exec("rm -f -- " + shlex.quote(remote_part), timeout=30)
        raise CloudWorkerError("音轨连续 3 次未通过文件大小与 SHA-256 校验，已拒绝进入识别阶段")

    def _download(self, remote_path: str, local_path: Path):
        if not self.sftp:
            raise CloudWorkerError("云节点文件通道尚未连接")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.sftp.get(remote_path, str(local_path))
        except FileNotFoundError as exc:
            raise CloudWorkerError(
                f"云节点没有生成预期结果文件：{remote_path}。"
                "远程运算可能提前退出，请查看该任务在此错误之前的云端日志"
            ) from exc

    def stage_job_audio(self, job_id: str, audio_path: Path) -> dict[str, object]:
        if not self.client:
            self.connect()
        self.remote_job_dir = posixpath.join(self.settings.remote_dir, "jobs", job_id)
        self._mkdir(self.remote_job_dir)
        return self._ensure_verified_audio(audio_path)

    def prepare_job(self, job_id: str, audio_path: Path, *, accuracy: bool = False):
        if not self.client:
            self.connect()
        self.remote_job_dir = posixpath.join(self.settings.remote_dir, "jobs", job_id)
        if self.settings.auto_setup:
            self.logger("检查并安装云节点运算依赖")
            if accuracy:
                self.bootstrap()
                self.bootstrap_accuracy()
            else:
                self.bootstrap()
        self._mkdir(posixpath.join(self.remote_job_dir, "studio"))
        for local, remote in (
            (ROOT / "asr_stage.py", posixpath.join(self.remote_job_dir, "asr_stage.py")),
            (ROOT / "large_review.py", posixpath.join(self.remote_job_dir, "large_review.py")),
            (ROOT / "audio_event_gate.py", posixpath.join(self.remote_job_dir, "audio_event_gate.py")),
            (ROOT / "studio" / "__init__.py", posixpath.join(self.remote_job_dir, "studio", "__init__.py")),
            (ROOT / "studio" / "languages.py", posixpath.join(self.remote_job_dir, "studio", "languages.py")),
        ):
            self._upload(local, remote, f"同步 {local.name}")
        self._ensure_verified_audio(audio_path)

        if accuracy:
            for name in (
                "ensemble_common.py",
                "qwen_primary_stage.py",
                "cohere_review_stage.py",
                "whisper_conflict_vote.py",
                "qwen_align_stage.py",
            ):
                self._upload(ROOT / name, posixpath.join(self.remote_job_dir, name), f"同步 {name}")

    def run_accuracy_asr(
        self,
        events_path: Path,
        local_workdir: Path,
        *,
        language: str,
        speech_threshold: float,
        nonlexical_factor: float,
    ):
        remote_work = posixpath.join(self.remote_job_dir, "accuracy_ensemble")
        self._mkdir(remote_work)
        remote_events = posixpath.join(remote_work, "events.json")
        primary = posixpath.join(remote_work, "qwen_primary.json")
        reviewed = posixpath.join(remote_work, "cohere_reviewed.json")
        voted = posixpath.join(remote_work, "ensemble_voted.json")
        audit = posixpath.join(remote_work, "ensemble_audit.json")
        final = posixpath.join(remote_work, "source_sentences.json")
        self._upload(events_path, remote_events, "上传多模型识别分段")
        models = self.settings.model_dir
        qwen_python = posixpath.join(models, "envs", "qwen", "bin", "python")
        cohere_python = posixpath.join(models, "envs", "cohere", "bin", "python")
        whisper_python = posixpath.join(self.settings.remote_dir, ".venv", "bin", "python")
        scripts = self.remote_job_dir
        audio = posixpath.join(self.remote_job_dir, "audio.flac")

        commands = [
            [
                qwen_python, posixpath.join(scripts, "qwen_primary_stage.py"), audio,
                "--events", remote_events, "--output", primary,
                "--model", posixpath.join(models, "weights", "Qwen3-ASR-1.7B"),
                "--language", language, "--speech-threshold", str(speech_threshold),
                "--nonlexical-factor", str(nonlexical_factor), "--batch-size", "4",
            ],
            [
                cohere_python, posixpath.join(scripts, "cohere_review_stage.py"), audio,
                "--input", primary, "--output", reviewed,
                "--model", posixpath.join(models, "weights", "cohere-transcribe-03-2026"),
                "--language", language, "--batch-size", "4",
            ],
            [
                whisper_python, posixpath.join(scripts, "whisper_conflict_vote.py"), audio,
                "--input", reviewed, "--output", voted, "--audit", audit,
                "--model", posixpath.join(models, "weights", "whisper", "large-v3.pt"),
                "--language", language,
            ],
            [
                qwen_python, posixpath.join(scripts, "qwen_align_stage.py"), audio,
                "--input", voted, "--output", final,
                "--model", posixpath.join(models, "weights", "Qwen3-ForcedAligner-0.6B"),
                "--language", language, "--batch-size", "8",
            ],
        ]
        for index, args in enumerate(commands, 1):
            self.logger(f"最高精度识别阶段 {index}/4")
            command = "env OMP_NUM_THREADS=4 " + " ".join(shlex.quote(value) for value in args)
            self._exec(command, controllable=True)
        self._download(final, local_workdir / "source_sentences.json")
        self._download(audit, local_workdir / "ensemble_audit.json")

    def run_event_gate(self, vad_path: Path, local_events_path: Path):
        remote_work = posixpath.join(self.remote_job_dir, "event_gate")
        self._mkdir(remote_work)
        remote_vad = posixpath.join(remote_work, "vad.json")
        remote_events = posixpath.join(remote_work, "events.json")
        self._upload(vad_path, remote_vad, "上传 VAD 分段")
        python = posixpath.join(self.settings.remote_dir, ".venv", "bin", "python")
        args = [
            "env", "OMP_NUM_THREADS=4",
            python,
            posixpath.join(self.remote_job_dir, "audio_event_gate.py"),
            posixpath.join(self.remote_job_dir, "audio.flac"),
            "--vad", remote_vad,
            "--output", remote_events,
        ]
        self._exec(" ".join(shlex.quote(value) for value in args), controllable=True)
        self._download(remote_events, local_events_path)

    def run_asr(
        self,
        events_path: Path,
        local_workdir: Path,
        *,
        label: str,
        model: str,
        language: str,
        speech_threshold: float,
        nonlexical_factor: float,
    ):
        remote_work = posixpath.join(self.remote_job_dir, label)
        self._mkdir(remote_work)
        remote_events = posixpath.join(remote_work, "events.json")
        self._upload(events_path, remote_events, "上传识别分段")
        python = posixpath.join(self.settings.remote_dir, ".venv", "bin", "python")
        args = [
            "env", "OMP_NUM_THREADS=4",
            python,
            posixpath.join(self.remote_job_dir, "asr_stage.py"),
            posixpath.join(self.remote_job_dir, "audio.flac"),
            "--events", remote_events,
            "--workdir", remote_work,
            "--model", model,
            "--language", language,
            "--speech-threshold", str(speech_threshold),
            "--nonlexical-factor", str(nonlexical_factor),
        ]
        self._exec(" ".join(shlex.quote(value) for value in args), controllable=True)
        self._download(
            posixpath.join(remote_work, "source_sentences.json"),
            local_workdir / "source_sentences.json",
        )

    def run_review(
        self,
        source_path: Path,
        local_workdir: Path,
        *,
        label: str,
        model: str,
        language: str,
    ):
        remote_work = posixpath.join(self.remote_job_dir, label)
        self._mkdir(remote_work)
        remote_source = posixpath.join(remote_work, "source_sentences.json")
        self._upload(source_path, remote_source, "上传复核文本")
        python = posixpath.join(self.settings.remote_dir, ".venv", "bin", "python")
        args = [
            "env", "OMP_NUM_THREADS=4",
            python,
            posixpath.join(self.remote_job_dir, "large_review.py"),
            posixpath.join(self.remote_job_dir, "audio.flac"),
            "--medium", remote_source,
            "--workdir", remote_work,
            "--model", model,
            "--language", language,
        ]
        self._exec(" ".join(shlex.quote(value) for value in args), controllable=True)
        for name in ("source_final.json", "model_comparison.json"):
            self._download(posixpath.join(remote_work, name), local_workdir / name)

    def cleanup_job(self):
        if self.remote_job_dir:
            self._exec("rm -rf -- " + shlex.quote(self.remote_job_dir), timeout=60)
            self.remote_job_dir = ""
