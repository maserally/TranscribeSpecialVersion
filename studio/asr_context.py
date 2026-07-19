from __future__ import annotations

from typing import Any


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    start = max(float(left.get("start", 0)), float(right.get("start", 0)))
    end = min(float(left.get("end", 0)), float(right.get("end", 0)))
    duration = max(0.001, float(left.get("end", 0)) - float(left.get("start", 0)))
    return max(0.0, end - start) / duration


def attach_asr_reviews(
    rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach verifier disagreement as translation context without changing timing."""
    enriched: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        candidates = [item for item in comparisons if _overlap(row, item) >= 0.5]
        if candidates:
            best = max(candidates, key=lambda item: _overlap(row, item))
            primary = str(best.get("large_v3", row.get("source", ""))).strip()
            alternative = str(best.get("medium", "")).strip()
            similarity = float(best.get("similarity", 1.0))
            row["asr_review"] = {
                "primary": primary,
                "alternative": alternative,
                "similarity": similarity,
                "disagreement": bool(alternative and similarity < 0.60),
            }
        enriched.append(row)
    return enriched
