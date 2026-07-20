from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel

from ensemble_common import review_reasons, select_windows


LANGUAGE_NAMES = {"ja": "Japanese", "ko": "Korean"}


def audio_slice(audio, sample_rate: int, start: float, end: float):
    return audio[int(start * sample_rate) : int(end * sample_rate)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("media")
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", choices=("ja", "ko"), required=True)
    parser.add_argument("--speech-threshold", type=float, default=0.10)
    parser.add_argument("--nonlexical-factor", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--event-gated", action="store_true")
    args = parser.parse_args()

    events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    audio, sample_rate = sf.read(args.media, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    windows = select_windows(
        events,
        args.speech_threshold,
        args.nonlexical_factor,
        duration=len(audio) / sample_rate,
        full_coverage=not args.event_gated,
    )

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=max(1, args.batch_size),
        max_new_tokens=384,
    )
    output_path = Path(args.output)
    output: list[dict] = []
    if output_path.exists():
        try:
            cached = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and all(
                isinstance(row, dict) and "window_index" in row for row in cached
            ):
                output = cached
        except (OSError, json.JSONDecodeError, TypeError):
            output = []
    completed = {int(row["window_index"]) for row in output if "window_index" in row}
    pending = [row for row in windows if int(row.get("window_index", -1)) not in completed]
    if completed:
        print(f"Qwen3-ASR resumed={len(completed)} pending={len(pending)}", flush=True)
    language = LANGUAGE_NAMES[args.language]
    for offset in range(0, len(pending), max(1, args.batch_size)):
        batch = pending[offset : offset + max(1, args.batch_size)]
        clips = [(audio_slice(audio, sample_rate, row["start"], row["end"]), sample_rate) for row in batch]
        results = model.transcribe(audio=clips, language=[language] * len(clips))
        for row, result in zip(batch, results):
            candidate = {
                **row,
                "qwen_source": str(result.text or "").strip(),
                "detected_language": str(result.language or language),
            }
            candidate["review_reasons"] = review_reasons(candidate)
            candidate["needs_review"] = bool(candidate["review_reasons"])
            output.append(candidate)
        output.sort(key=lambda row: int(row.get("window_index", 0)))
        done = len(completed) + min(offset + len(batch), len(pending))
        print(f"Qwen3-ASR {done}/{len(windows)}", flush=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Qwen3-ASR windows={len(output)} low_confidence={sum(bool(x['needs_review']) for x in output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
