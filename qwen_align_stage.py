from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import soundfile as sf
import torch
from qwen_asr import Qwen3ForcedAligner


LANGUAGE_NAMES = {"ja": "Japanese", "ko": "Korean"}
SENTENCE_END = re.compile(r"[。！？!?]$")


def split_tokens(
    tokens,
    window_start: float,
    language: str,
    winner: str,
    *,
    core_start: float,
    core_end: float,
    window_index: int,
    confidence: str,
) -> list[dict]:
    sentences: list[dict] = []
    current = []

    def flush():
        nonlocal current
        if not current:
            return
        pieces = [str(token.text).strip() for token in current if str(token.text).strip()]
        text = (" ".join(pieces) if language == "ko" else "".join(pieces)).strip()
        if text:
            sentences.append(
                {
                    "start": round(window_start + float(current[0].start_time), 3),
                    "end": round(window_start + float(current[-1].end_time), 3),
                    "source": text,
                    "asr_engine": "qwen-cohere-whisper-ensemble",
                    "ensemble_winner": winner,
                    "ensemble_confidence": confidence,
                    "window_index": window_index,
                }
            )
        current = []

    token_items = getattr(tokens, "items", tokens)
    for token in token_items:
        midpoint = window_start + (
            float(token.start_time) + float(token.end_time)
        ) / 2
        if midpoint < core_start or midpoint >= core_end:
            continue
        if current:
            duration = float(current[-1].end_time) - float(current[0].start_time)
            chars = sum(len(str(item.text).strip()) for item in current)
            gap = float(token.start_time) - float(current[-1].end_time)
            if duration >= 7.0 or chars >= 34 or gap > 0.85:
                flush()
        current.append(token)
        if SENTENCE_END.search(str(token.text).strip()):
            flush()
    flush()
    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("media")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", choices=("ja", "ko"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    audio, sample_rate = sf.read(args.media, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    model = Qwen3ForcedAligner.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    output_path = Path(args.output)
    sentences: list[dict] = []
    if output_path.exists():
        try:
            cached = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and all(
                isinstance(row, dict) and "window_index" in row for row in cached
            ):
                sentences = cached
        except (OSError, json.JSONDecodeError, TypeError):
            sentences = []
    completed = {
        int(row["window_index"]) for row in sentences if "window_index" in row
    }
    active_rows = [row for row in rows if str(row.get("final_source") or "").strip()]
    pending_rows = [
        row for row in active_rows
        if int(row.get("window_index", -1)) not in completed
    ]
    if completed:
        print(f"Qwen3-ForcedAligner resumed={len(completed)} pending={len(pending_rows)}", flush=True)
    batch_size = max(1, args.batch_size)
    for offset in range(0, len(pending_rows), batch_size):
        batch = pending_rows[offset : offset + batch_size]
        clips = [
            (audio[int(row["start"] * sample_rate) : int(row["end"] * sample_rate)], sample_rate)
            for row in batch
        ]
        texts = [str(row["final_source"]).strip() for row in batch]
        aligned_batch = model.align(
            audio=clips,
            text=texts,
            language=[LANGUAGE_NAMES[args.language]] * len(batch),
        )
        for row, text, aligned in zip(batch, texts, aligned_batch):
            aligned_rows = split_tokens(
                aligned,
                float(row["start"]),
                args.language,
                str(row.get("ensemble_winner", "qwen")),
                core_start=float(row.get("core_start", row["start"])),
                core_end=float(row.get("core_end", row["end"])),
                window_index=int(row.get("window_index", 0)),
                confidence=str(row.get("ensemble_confidence", "unknown")),
            )
            if not aligned_rows:
                aligned_rows = [
                    {
                        "start": round(float(row.get("core_start", row["start"])), 3),
                        "end": round(float(row.get("core_end", row["end"])), 3),
                        "source": text,
                        "asr_engine": "qwen-cohere-whisper-ensemble",
                        "ensemble_winner": row.get("ensemble_winner", "qwen"),
                        "ensemble_confidence": row.get("ensemble_confidence", "unknown"),
                        "window_index": int(row.get("window_index", 0)),
                        "alignment_fallback": True,
                    }
                ]
            sentences.extend(aligned_rows)
        print(
            f"Qwen3-ForcedAligner {len(completed) + min(offset + len(batch), len(pending_rows))}/{len(active_rows)}",
            flush=True,
        )
        output_path.write_text(
            json.dumps(sentences, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    sentences.sort(key=lambda item: (item["start"], item["end"]))
    output_path.write_text(json.dumps(sentences, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"aligned sentences={len(sentences)}", flush=True)


if __name__ == "__main__":
    main()
