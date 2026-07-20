from __future__ import annotations

import re
from difflib import SequenceMatcher


PUNCT_RE = re.compile(r"[\s\u3000、。！？!?…・,.~〜～‘’“”\"'()（）\[\]【】]+")


def normalize_transcript(text: str) -> str:
    return PUNCT_RE.sub("", str(text or "")).casefold()


def transcript_similarity(left: str, right: str) -> float:
    a = normalize_transcript(left)
    b = normalize_transcript(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def repetition_ratio(text: str) -> float:
    compact = normalize_transcript(text)
    if len(compact) < 6:
        return 0.0
    best = 0.0
    for size in range(1, min(12, len(compact) // 2) + 1):
        for start in range(min(size, len(compact))):
            unit = compact[start : start + size]
            if not unit:
                continue
            pos = start
            repeats = 0
            while compact.startswith(unit, pos):
                repeats += 1
                pos += size
            if repeats >= 3:
                best = max(best, (pos - start) / len(compact))
    return best


def review_reasons(row: dict) -> list[str]:
    text = str(row.get("qwen_source") or "").strip()
    duration = max(0.0, float(row.get("end", 0)) - float(row.get("start", 0)))
    speech_score = float(row.get("speech_score", 0))
    nonlexical_score = float(row.get("nonlexical_score", 0))
    reasons: list[str] = []
    if not normalize_transcript(text):
        reasons.append("empty")
    if speech_score < 0.42:
        reasons.append("weak_speech")
    if nonlexical_score >= 0.20 or (
        nonlexical_score > 0 and speech_score < nonlexical_score * 2.2
    ):
        reasons.append("music_or_noise_masking")
    if duration >= 2.4 and len(normalize_transcript(text)) <= 2:
        reasons.append("too_short_for_window")
    if repetition_ratio(text) >= 0.55:
        reasons.append("repetition")
    return reasons


def needs_third_vote(qwen_text: str, cohere_text: str, threshold: float = 0.72) -> bool:
    qwen = normalize_transcript(qwen_text)
    cohere = normalize_transcript(cohere_text)
    if not qwen and not cohere:
        return False
    if not qwen or not cohere:
        return True
    return transcript_similarity(qwen_text, cohere_text) < threshold


def choose_consensus(qwen_text: str, cohere_text: str, whisper_text: str) -> tuple[str, str, dict]:
    candidates = {
        "qwen": str(qwen_text or "").strip(),
        "cohere": str(cohere_text or "").strip(),
        "whisper": str(whisper_text or "").strip(),
    }
    nonempty = {key: value for key, value in candidates.items() if normalize_transcript(value)}
    if not nonempty:
        return "", "none", {"qwen_cohere": 0.0, "qwen_whisper": 0.0, "cohere_whisper": 0.0}
    if len(nonempty) == 1:
        key, value = next(iter(nonempty.items()))
        return value, key, {"qwen_cohere": 0.0, "qwen_whisper": 0.0, "cohere_whisper": 0.0}

    similarities = {
        "qwen_cohere": transcript_similarity(candidates["qwen"], candidates["cohere"]),
        "qwen_whisper": transcript_similarity(candidates["qwen"], candidates["whisper"]),
        "cohere_whisper": transcript_similarity(candidates["cohere"], candidates["whisper"]),
    }
    scores = {
        "qwen": similarities["qwen_cohere"] + similarities["qwen_whisper"] + 0.02,
        "cohere": similarities["qwen_cohere"] + similarities["cohere_whisper"],
        "whisper": similarities["qwen_whisper"] + similarities["cohere_whisper"],
    }
    winner = max(nonempty, key=lambda key: scores[key])
    return candidates[winner], winner, similarities


def _scores_for_range(events: list[dict], start: float, end: float) -> tuple[float, float]:
    overlaps = [
        row for row in events
        if float(row.get("end", 0)) > start and float(row.get("start", 0)) < end
    ]
    if not overlaps:
        return 0.0, 0.0
    return (
        max(float(row.get("speech_score", 0)) for row in overlaps),
        max(float(row.get("nonlexical_score", 0)) for row in overlaps),
    )


def full_coverage_windows(
    events: list[dict],
    duration: float,
    *,
    core_seconds: float = 16.0,
    context_seconds: float = 2.0,
) -> list[dict]:
    """Partition all audio into cores with context; event scores never drop audio."""
    duration = max(0.0, float(duration))
    core_seconds = max(4.0, float(core_seconds))
    context_seconds = max(0.0, float(context_seconds))
    windows: list[dict] = []
    core_start = 0.0
    index = 0
    while core_start < duration:
        core_end = min(duration, core_start + core_seconds)
        start = max(0.0, core_start - context_seconds)
        end = min(duration, core_end + context_seconds)
        speech_score, nonlexical_score = _scores_for_range(events, core_start, core_end)
        windows.append(
            {
                "window_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "core_start": round(core_start, 3),
                "core_end": round(core_end, 3),
                "speech_score": speech_score,
                "nonlexical_score": nonlexical_score,
                "coverage_origin": "full_audio",
            }
        )
        index += 1
        core_start = core_end
    return windows


def select_windows(
    events: list[dict],
    speech_threshold: float,
    nonlexical_factor: float,
    *,
    duration: float | None = None,
    full_coverage: bool = False,
) -> list[dict]:
    if full_coverage:
        if duration is None:
            duration = max((float(row.get("end", 0)) for row in events), default=0.0)
        return full_coverage_windows(events, duration)
    selected = [
        row for row in events
        if float(row.get("speech_score", 0)) >= speech_threshold
        and float(row.get("speech_score", 0))
        > float(row.get("nonlexical_score", 0)) * nonlexical_factor
    ]
    merged: list[dict] = []
    for row in selected:
        start = max(0.0, float(row["start"]) - 0.22)
        end = float(row["end"]) + 0.28
        if merged and start <= merged[-1]["end"] + 0.32 and end - merged[-1]["start"] <= 24.0:
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["speech_score"] = max(merged[-1]["speech_score"], float(row["speech_score"]))
            merged[-1]["nonlexical_score"] = max(
                merged[-1]["nonlexical_score"], float(row.get("nonlexical_score", 0))
            )
        else:
            merged.append(
                {
                    "start": start,
                    "end": end,
                    "speech_score": float(row["speech_score"]),
                    "nonlexical_score": float(row.get("nonlexical_score", 0)),
                }
            )
    return merged
