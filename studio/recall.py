from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .quality import find_gaps


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


def accepted_recovery_rows(
    recovery_final: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    consensus_threshold: float,
):
    accepted = []
    for row in recovery_final:
        if row.get("source") == "medium_fallback":
            continue
        similarities = [
            item["similarity"]
            for item in comparisons
            if min(row["end"], item["end"]) - max(row["start"], item["start"]) > 0.1
        ]
        if similarities and max(similarities) >= consensus_threshold:
            item = dict(row)
            item["source"] = "gap_recovery_consensus"
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
