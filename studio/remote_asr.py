from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np
import whisper

from asr_stage import HALLUCINATIONS, pack_windows, select_windows

from .providers import OpenAICompatibleProvider, supports_segment_timestamps
from .schemas import ProviderSettings


def _write_wav(path: Path, audio: np.ndarray):
    pcm = np.clip(audio, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(pcm.tobytes())


def _mapping(midpoint: float, mappings: list[dict[str, Any]]):
    return next(
        (m for m in mappings if m["packed_start"] - 0.08 <= midpoint <= m["packed_end"] + 0.08),
        None,
    )


def _split_segment(row: dict[str, Any]):
    parts = [x for x in re.split(r"(?<=[。！？.!?])", row["source"]) if x.strip()]
    if len(parts) <= 1:
        return [row]
    total = sum(max(1, len(x)) for x in parts)
    cursor = row["start"]
    duration = row["end"] - row["start"]
    output = []
    for index, part in enumerate(parts):
        end = row["end"] if index == len(parts) - 1 else cursor + duration * len(part) / total
        output.append({**row, "start": cursor, "end": end, "source": part.strip()})
        cursor = end
    return output


def _remote_packs(windows: list[dict[str, Any]], timestamped: bool):
    if timestamped:
        return pack_windows(windows, max_audio=24.0, separator=0.8)
    # JSON-only transcription models need one VAD window per request so the
    # whole response can inherit that window's original timeline.
    return [pack_windows([window], max_audio=30.0, separator=0.0)[0] for window in windows]


def run_remote_asr(
    media: Path,
    events: list[dict[str, Any]],
    settings: ProviderSettings,
    workdir: Path,
    speech_threshold: float,
    nonlexical_factor: float,
    language: str,
    progress: Callable[[int, int, str], None] | None = None,
):
    provider = OpenAICompatibleProvider(settings.base_url, settings.api_key, timeout=900)
    windows = select_windows(events, speech_threshold, nonlexical_factor)
    timestamped = supports_segment_timestamps(settings.model)
    packs = _remote_packs(windows, timestamped)
    audio = whisper.load_audio(str(media))
    rows = []
    temp_wav = workdir / "remote_asr_chunk.wav"
    for batch_index, pack in enumerate(packs, 1):
        clip = np.zeros(int(pack["duration"] * 16000), dtype=np.float32)
        for mapping in pack["mappings"]:
            source = audio[
                int(mapping["original_start"] * 16000) : int(mapping["original_end"] * 16000)
            ]
            start = int(mapping["packed_start"] * 16000)
            end = min(len(clip), start + len(source))
            clip[start:end] = source[: end - start]
        _write_wav(temp_wav, clip)
        result = provider.transcribe(settings.model, temp_wav, language)
        segments = result.get("segments")
        if not timestamped:
            text = str(result.get("text", "")).strip()
            mapping = pack["mappings"][0]
            if text and not any(x in text for x in HALLUCINATIONS.get(language, ())):
                rows.extend(
                    _split_segment(
                        {
                            "start": round(mapping["original_start"], 3),
                            "end": round(mapping["original_end"], 3),
                            "source": text,
                            "mean_word_probability": 0.9,
                            "asr_source": "openai_compatible_window_timing",
                        }
                    )
                )
        elif not isinstance(segments, list):
            raise RuntimeError(
                "该 OpenAI 兼容转写服务没有返回 verbose_json.segments，无法恢复原视频时间轴"
            )
        for segment in segments or []:
            text = str(segment.get("text", "")).strip()
            midpoint = (float(segment.get("start", 0)) + float(segment.get("end", 0))) / 2
            mapping = _mapping(midpoint, pack["mappings"])
            if not mapping or not text or any(
                x in text for x in HALLUCINATIONS.get(language, ())
            ):
                continue
            start = mapping["original_start"] + max(0, float(segment["start"]) - mapping["packed_start"])
            end = mapping["original_start"] + min(
                mapping["original_end"] - mapping["original_start"],
                float(segment["end"]) - mapping["packed_start"],
            )
            if end > start:
                rows.extend(
                    _split_segment(
                        {
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "source": text,
                            "mean_word_probability": 0.9,
                            "asr_source": "openai_compatible",
                        }
                    )
                )
        if progress:
            progress(batch_index, len(packs), f"远程转写批次 {batch_index}/{len(packs)}")
    rows.sort(key=lambda x: (x["start"], x["end"]))
    (workdir / "source_sentences.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (workdir / "source_final.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows
