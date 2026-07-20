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


def split_tokens(tokens, window_start: float, language: str, winner: str) -> list[dict]:
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
                }
            )
        current = []

    for token in tokens:
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
    sentences: list[dict] = []
    active_rows = [row for row in rows if str(row.get("final_source") or "").strip()]
    batch_size = max(1, args.batch_size)
    for offset in range(0, len(active_rows), batch_size):
        batch = active_rows[offset : offset + batch_size]
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
            )
            if not aligned_rows:
                aligned_rows = [
                    {
                        "start": round(float(row["start"]), 3),
                        "end": round(float(row["end"]), 3),
                        "source": text,
                        "asr_engine": "qwen-cohere-whisper-ensemble",
                        "ensemble_winner": row.get("ensemble_winner", "qwen"),
                        "alignment_fallback": True,
                    }
                ]
            sentences.extend(aligned_rows)
        print(
            f"Qwen3-ForcedAligner {min(offset + len(batch), len(active_rows))}/{len(active_rows)}",
            flush=True,
        )
        Path(args.output).write_text(
            json.dumps(sentences, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    sentences.sort(key=lambda item: (item["start"], item["end"]))
    Path(args.output).write_text(json.dumps(sentences, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"aligned sentences={len(sentences)}", flush=True)


if __name__ == "__main__":
    main()
