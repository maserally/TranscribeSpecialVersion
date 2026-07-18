from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


PROFILE_SETTINGS = {
    "precision": {
        "speech_threshold": 0.20,
        "nonlexical_factor": 1.35,
        "recovery_threshold": 0.12,
        "consensus_threshold": 0.65,
    },
    "balanced": {
        "speech_threshold": 0.15,
        "nonlexical_factor": 1.20,
        "recovery_threshold": 0.08,
        "consensus_threshold": 0.52,
    },
    "recall": {
        "speech_threshold": 0.10,
        "nonlexical_factor": 1.05,
        "recovery_threshold": 0.045,
        "consensus_threshold": 0.40,
    },
}


def strip_chinese_periods(text: str) -> str:
    return re.sub(r"。|\.(?=\s*$)", "", text).strip()


def publish_text(text: str) -> str:
    text = text.replace("【听不清】", "").replace("[听不清]", "")
    text = text.replace("【地点名不明】", "").replace("【场所名不明】", "")
    text = text.replace("【人名不明】", "有人")
    text = re.sub(r"。|\.(?=\s*$)", "", text)
    return re.sub(r"\s+", " ", text).strip(" ，、")


def finalize_cues(
    cues: list[dict[str, Any]],
    *,
    min_duration: float = 0.85,
    remove_periods: bool = True,
    publish: bool = False,
) -> list[dict[str, Any]]:
    rows = [dict(x) for x in sorted(cues, key=lambda x: (x["start"], x["end"]))]
    if publish:
        for row in rows:
            row["zh"] = publish_text(row.get("zh", ""))
        rows = [x for x in rows if x.get("zh", "").strip()]
    if remove_periods:
        for row in rows:
            row["zh"] = strip_chinese_periods(row.get("zh", ""))

    # Merge sub-second adjacent cues when that is safer than flashing text.
    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(rows):
        row = dict(rows[index])
        duration = row["end"] - row["start"]
        if duration < min_duration and index + 1 < len(rows):
            nxt = rows[index + 1]
            gap = nxt["start"] - row["end"]
            combined_len = len(row.get("zh", "")) + len(nxt.get("zh", ""))
            if gap <= 0.18 and combined_len <= 24:
                row["end"] = nxt["end"]
                row["ja"] = (row.get("ja", "") + nxt.get("ja", "")).strip()
                row["zh"] = (row.get("zh", "") + "，" + nxt.get("zh", "")).strip("，")
                index += 1
        merged.append(row)
        index += 1

    for index, row in enumerate(merged):
        next_start = merged[index + 1]["start"] if index + 1 < len(merged) else row["end"] + 2
        if row["end"] - row["start"] < min_duration:
            row["end"] = min(next_start - 0.04, row["start"] + min_duration)
        if index + 1 < len(merged) and row["end"] > next_start - 0.04:
            row["end"] = max(row["start"] + 0.30, next_start - 0.04)
        row["id"] = index + 1
    return merged


def find_gaps(cues: list[dict[str, Any]], duration: float, threshold: float = 30.0):
    gaps = []
    cursor = 0.0
    for cue in sorted(cues, key=lambda x: x["start"]):
        if cue["start"] - cursor >= threshold:
            gaps.append({"start": cursor, "end": cue["start"], "duration": cue["start"] - cursor})
        cursor = max(cursor, cue["end"])
    if duration - cursor >= threshold:
        gaps.append({"start": cursor, "end": duration, "duration": duration - cursor})
    return gaps


def quality_summary(cues: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    durations = [x["end"] - x["start"] for x in cues]
    gaps = find_gaps(cues, duration)
    return {
        "cue_count": len(cues),
        "display_seconds": round(sum(durations), 3),
        "under_085_seconds": sum(x < 0.85 for x in durations),
        "exact_two_seconds": sum(abs(x - 2.0) < 0.01 for x in durations),
        "overlaps": sum(cues[i]["end"] > cues[i + 1]["start"] for i in range(len(cues) - 1)),
        "placeholders": sum("【" in x.get("zh", "") for x in cues),
        "chinese_periods": sum("。" in x.get("zh", "") for x in cues),
        "long_gaps": gaps,
        "longest_gap": round(max((x["duration"] for x in gaps), default=0.0), 3),
    }
