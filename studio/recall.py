from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .quality import find_gaps


def _subtract_intervals(start: float, end: float, blockers: list[tuple[float, float]]):
    pieces = [(start, end)]
    for block_start, block_end in blockers:
        next_pieces = []
        for piece_start, piece_end in pieces:
            if block_end <= piece_start or block_start >= piece_end:
                next_pieces.append((piece_start, piece_end))
                continue
            if block_start > piece_start:
                next_pieces.append((piece_start, min(piece_end, block_start)))
            if block_end < piece_end:
                next_pieces.append((max(piece_start, block_end), piece_end))
        pieces = next_pieces
    return pieces


def vad_fallback_events_for_gaps(
    vad_segments: list[dict[str, Any]],
    gaps: list[dict[str, float]],
    existing_events: list[dict[str, Any]],
    min_duration: float = 0.24,
):
    """Return VAD-confirmed pieces in long gaps not already reviewed by the event gate."""
    blockers = sorted((float(x["start"]), float(x["end"])) for x in existing_events)
    output = []
    for segment in vad_segments:
        for gap in gaps:
            start = max(float(segment["start"]), float(gap["start"]))
            end = min(float(segment["end"]), float(gap["end"]))
            if end - start < min_duration:
                continue
            for piece_start, piece_end in _subtract_intervals(start, end, blockers):
                if piece_end - piece_start < min_duration:
                    continue
                output.append(
                    {
                        "start": round(piece_start, 3),
                        "end": round(piece_end, 3),
                        "speech_score": 1.0,
                        "nonlexical_score": 0.0,
                        "speech_margin": 1.0,
                        "source": "vad_gap_fallback",
                    }
                )
    output.sort(key=lambda x: (x["start"], x["end"]))
    merged = []
    for row in output:
        if merged and row["start"] <= merged[-1]["end"] + 0.05:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
        else:
            merged.append(dict(row))
    return merged


def filter_events_for_gaps(
    events: list[dict[str, Any]],
    gaps: list[dict[str, float]],
    recovery_threshold: float,
    nonlexical_factor: float,
):
    return [
        event
        for event in events
        if event["speech_score"] >= recovery_threshold
        and event["speech_score"] > event["nonlexical_score"] * max(1.0, nonlexical_factor - 0.1)
        and any(event["end"] > gap["start"] and event["start"] < gap["end"] for gap in gaps)
    ]


def filter_events_for_uncovered_speech(
    events: list[dict[str, Any]],
    recognized: list[dict[str, Any]],
    recovery_threshold: float,
    nonlexical_factor: float,
):
    """Find short, low-confidence speech candidates that normal long-gap recovery misses."""
    output = []
    for event in events:
        if event["speech_score"] < recovery_threshold:
            continue
        if event["speech_score"] <= event["nonlexical_score"] * max(1.0, nonlexical_factor - 0.2):
            continue
        if any(
            min(float(event["end"]), float(row["end"]))
            - max(float(event["start"]), float(row["start"]))
            > 0.2
            for row in recognized
        ):
            continue
        candidate = dict(event)
        candidate["source"] = "music_recovery_candidate"
        output.append(candidate)
    return output


def accepted_recovery_rows(
    recovery_final: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    consensus_threshold: float,
    recovery_source: str = "gap_recovery_consensus",
):
    accepted = []
    for row in recovery_final:
        if row.get("review_source") == "medium_fallback":
            continue
        similarities = [
            item["similarity"]
            for item in comparisons
            if min(row["end"], item["end"]) - max(row["start"], item["start"]) > 0.1
        ]
        if similarities and max(similarities) >= consensus_threshold:
            item = dict(row)
            item["recovery_source"] = recovery_source
            accepted.append(item)
    return accepted


def merge_recovery(primary: list[dict[str, Any]], recovery: list[dict[str, Any]]):
    merged = list(primary)
    for row in recovery:
        overlap = any(
            min(row["end"], existing["end"]) - max(row["start"], existing["start"]) > 0.2
            for existing in merged
        )
        if not overlap:
            merged.append(row)
    merged.sort(key=lambda x: (x["start"], x["end"]))
    return merged


def save_gap_audit(path: Path, cues: list[dict[str, Any]], duration: float):
    gaps = find_gaps(cues, duration)
    path.write_text(json.dumps(gaps, ensure_ascii=False, indent=2), encoding="utf-8")
    return gaps
