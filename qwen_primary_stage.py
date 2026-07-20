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
    args = parser.parse_args()

    events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    windows = select_windows(events, args.speech_threshold, args.nonlexical_factor)
    audio, sample_rate = sf.read(args.media, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=max(1, args.batch_size),
        max_new_tokens=384,
    )
    output: list[dict] = []
    language = LANGUAGE_NAMES[args.language]
    for offset in range(0, len(windows), max(1, args.batch_size)):
        batch = windows[offset : offset + max(1, args.batch_size)]
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
        print(f"Qwen3-ASR {min(offset + len(batch), len(windows))}/{len(windows)}", flush=True)
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Qwen3-ASR windows={len(output)} low_confidence={sum(bool(x['needs_review']) for x in output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
