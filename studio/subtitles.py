from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def srt_ts(seconds: float) -> str:
    millis = max(0, int(round(seconds * 1000)))
    h, rem = divmod(millis, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap_cn(text: str, width: int = 22) -> str:
    if len(text) <= width:
        return text
    candidates = [i + 1 for i, char in enumerate(text[: width + 5]) if char in "，、；：！？"]
    cut = candidates[-1] if candidates else width
    return text[:cut].rstrip() + "\n" + text[cut:].lstrip()


def write_subtitles(cues: list[dict[str, Any]], output_dir: Path, stem: str, suffix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    cn = output_dir / f"{stem}_中文字幕_{suffix}.srt"
    bi = output_dir / f"{stem}_中日双语_{suffix}.srt"
    ja = output_dir / f"{stem}_日文字幕_{suffix}.srt"
    data = output_dir / f"{stem}_字幕数据_{suffix}.json"
    data.write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
    with cn.open("w", encoding="utf-8-sig") as f_cn, \
         bi.open("w", encoding="utf-8-sig") as f_bi, \
         ja.open("w", encoding="utf-8-sig") as f_ja:
        for index, row in enumerate(cues, 1):
            timeline = f"{srt_ts(row['start'])} --> {srt_ts(row['end'])}"
            f_cn.write(f"{index}\n{timeline}\n{wrap_cn(row.get('zh', ''))}\n\n")
            f_bi.write(
                f"{index}\n{timeline}\n{wrap_cn(row.get('zh', ''))}\n{row.get('ja', '')}\n\n"
            )
            f_ja.write(f"{index}\n{timeline}\n{row.get('ja', '')}\n\n")
    return {"cn_srt": cn, "bilingual_srt": bi, "ja_srt": ja, "json": data}


def mux_soft_subtitles(media: Path, subtitle: Path, output: Path, title: str = "简体中文"):
    command = [
        "ffmpeg", "-hide_banner", "-y", "-i", str(media), "-i", str(subtitle),
        "-map", "0:v", "-map", "0:a?", "-map", "1:0",
        "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
        "-metadata:s:s:0", "language=zho", "-metadata:s:s:0", f"title={title}",
        "-disposition:s:0", "default", "-movflags", "+faststart", str(output),
    ]
    subprocess.run(command, check=True)
    return output


def mux_hard_subtitles(media: Path, subtitle: Path, output: Path):
    """Burn Chinese subtitles into the video with a stable ASCII filter path."""
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            prefix="burn_", suffix=".srt", dir=output.parent, delete=False
        ) as temporary:
            temporary_name = Path(temporary.name).name
        shutil.copyfile(subtitle, output.parent / temporary_name)
        style = (
            "FontName=Microsoft YaHei,FontSize=18,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,"
            "MarginV=28,Alignment=2"
        )
        command = [
            "ffmpeg", "-hide_banner", "-y", "-i", str(media),
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", f"subtitles=filename='{temporary_name}':charenc=UTF-8:force_style='{style}'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
        ]
        subprocess.run(command, cwd=output.parent, check=True)
        return output
    finally:
        if temporary_name:
            (output.parent / temporary_name).unlink(missing_ok=True)
