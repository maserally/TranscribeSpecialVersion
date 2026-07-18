import argparse
import json
from pathlib import Path

import torch
import whisper
from transformers import ASTFeatureExtractor, ASTForAudioClassification


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SPEECH_IDS = [0, 1, 2, 3, 4, 15]
NONLEXICAL_IDS = [14, 22, 24, 25, 26, 38, 39, 41, 42, 44, 45, 46]


def split_vad(vad_segments, max_duration=8.0):
    units = []
    for seg in vad_segments:
        start, end = float(seg["start"]), float(seg["end"])
        if end - start <= max_duration:
            units.append({"start": start, "end": end})
            continue
        cursor = start
        while cursor < end:
            unit_end = min(end, cursor + max_duration)
            units.append({"start": cursor, "end": unit_end})
            if unit_end >= end:
                break
            cursor = unit_end - 0.35
    return units


def classify(media, units, batch_size=12):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading audio...", flush=True)
    audio = whisper.load_audio(media)
    print("Loading AudioSet AST model...", flush=True)
    extractor = ASTFeatureExtractor()
    model = ASTForAudioClassification.from_pretrained(MODEL_NAME).to(device).eval()
    id2label = model.config.id2label
    rows = []

    for offset in range(0, len(units), batch_size):
        batch_units = units[offset : offset + batch_size]
        clips = []
        for unit in batch_units:
            start = max(0, int(unit["start"] * 16000))
            end = min(len(audio), int(unit["end"] * 16000))
            clips.append(audio[start:end])
        inputs = extractor(
            clips, sampling_rate=16000, return_tensors="pt", padding="max_length"
        )
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device)).logits
            probs = logits.sigmoid().cpu()
        for unit, scores in zip(batch_units, probs):
            speech = float(scores[SPEECH_IDS].max())
            nonlexical = float(scores[NONLEXICAL_IDS].max())
            top = torch.topk(scores, 5)
            row = {
                **unit,
                "speech_score": round(speech, 6),
                "nonlexical_score": round(nonlexical, 6),
                "speech_margin": round(speech - nonlexical, 6),
                "top_labels": [
                    {"label": id2label[int(i)], "score": round(float(v), 6)}
                    for v, i in zip(top.values, top.indices)
                ],
            }
            rows.append(row)
        print(f"Classified {min(offset + batch_size, len(units))}/{len(units)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--vad")
    ap.add_argument("--output", required=True)
    ap.add_argument("--times", help="Comma-separated centers for 10-second test windows")
    args = ap.parse_args()

    if args.times:
        units = [
            {"start": max(0, float(t) - 5), "end": float(t) + 5}
            for t in args.times.split(",")
        ]
    else:
        vad = json.loads(Path(args.vad).read_text(encoding="utf-8"))
        units = split_vad(vad)
    rows = classify(args.media, units)
    Path(args.output).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for row in rows if args.times else []:
        print(
            f"{row['start']:.1f}-{row['end']:.1f} "
            f"speech={row['speech_score']:.3f} nonlex={row['nonlexical_score']:.3f} "
            f"top={row['top_labels']}"
        )


if __name__ == "__main__":
    main()
