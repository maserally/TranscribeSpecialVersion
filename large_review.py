import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
import whisper

from asr_stage import pack_windows, rejection_reasons


def merge_review_windows(sentences):
    windows = []
    for sentence in sentences:
        start = max(0.0, sentence["start"] - 0.65)
        end = sentence["end"] + 0.65
        if windows and start <= windows[-1]["end"] + 0.65 and end - windows[-1]["start"] <= 24:
            windows[-1]["end"] = max(windows[-1]["end"], end)
        else:
            windows.append(
                {
                    "start": start,
                    "end": end,
                    "speech_scores": [1.0],
                    "nonlexical_scores": [0.0],
                }
            )
    return windows


def normalize(text):
    return re.sub(r"[\s、。！？!?…・〜～ー]", "", text)


def similarity(a, b):
    a, b = normalize(a), normalize(b)
    return SequenceMatcher(None, a, b).ratio() if a or b else 1.0


def mapping_for_midpoint(midpoint, mappings):
    return next(
        (
            m for m in mappings
            if m["packed_start"] - 0.06 <= midpoint <= m["packed_end"] + 0.06
        ),
        None,
    )


def split_text_segment(row):
    parts = [x for x in re.split(r"(?<=[。！？!?])", row["ja"]) if x.strip()]
    if len(parts) <= 1:
        return [row]
    total_chars = sum(max(1, len(normalize(x))) for x in parts)
    duration = row["end"] - row["start"]
    cursor = row["start"]
    output = []
    for index, part in enumerate(parts):
        fraction = max(1, len(normalize(part))) / total_chars
        end = row["end"] if index == len(parts) - 1 else cursor + duration * fraction
        output.append({**row, "start": round(cursor, 3), "end": round(end, 3), "ja": part.strip()})
        cursor = end
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--medium", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model", default="large-v3")
    args = ap.parse_args()
    workdir = Path(args.workdir)
    medium = json.loads(Path(args.medium).read_text(encoding="utf-8"))
    windows = merge_review_windows(medium)
    packs = pack_windows(windows, max_audio=26.0, separator=0.7)
    print(
        f"large-v3 review_windows={len(windows)} packed_batches={len(packs)} "
        f"audio_minutes={sum(x['end']-x['start'] for x in windows)/60:.2f}",
        flush=True,
    )
    audio = whisper.load_audio(args.media)
    model = whisper.load_model(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    large_rows = []

    for batch_index, pack in enumerate(packs, 1):
        clip = np.zeros(int(pack["duration"] * 16000), dtype=np.float32)
        for mapping in pack["mappings"]:
            source = audio[
                int(mapping["original_start"] * 16000) :
                int(mapping["original_end"] * 16000)
            ]
            target_start = int(mapping["packed_start"] * 16000)
            target_end = min(len(clip), target_start + len(source))
            clip[target_start:target_end] = source[: target_end - target_start]
        result = model.transcribe(
            clip,
            language="ja",
            task="transcribe",
            fp16=torch.cuda.is_available(),
            temperature=0,
            condition_on_previous_text=False,
            verbose=False,
        )
        for seg in result["segments"]:
            midpoint = (seg["start"] + seg["end"]) / 2
            mapping = mapping_for_midpoint(midpoint, pack["mappings"])
            if mapping is None:
                continue
            absolute_start = mapping["original_start"] + max(
                0.0, seg["start"] - mapping["packed_start"]
            )
            absolute_end = mapping["original_start"] + min(
                mapping["original_end"] - mapping["original_start"],
                seg["end"] - mapping["packed_start"],
            )
            row = {
                "start": round(max(mapping["original_start"], absolute_start), 3),
                "end": round(min(mapping["original_end"], absolute_end), 3),
                "ja": seg["text"].strip(),
                "avg_logprob": seg["avg_logprob"],
                "no_speech_prob": seg["no_speech_prob"],
                "compression_ratio": seg["compression_ratio"],
                "batch": batch_index,
            }
            row["rejection_reasons"] = rejection_reasons({**seg, "text": row["ja"]})
            if row["end"] > row["start"] and row["ja"]:
                large_rows.append(row)
        print(f"large-v3 {batch_index}/{len(packs)}", flush=True)

    accepted_large = []
    for row in large_rows:
        if not row["rejection_reasons"]:
            accepted_large.extend(split_text_segment(row))

    # Fill only high-confidence medium lines when large-v3 produced no overlapping text.
    filled = list(accepted_large)
    for row in medium:
        overlap = any(
            min(row["end"], other["end"]) - max(row["start"], other["start"]) > 0.25
            for other in accepted_large
        )
        if not overlap and row["mean_word_probability"] >= 0.78:
            filled.append({**row, "source": "medium_fallback"})
    filled.sort(key=lambda x: (x["start"], x["end"]))

    comparisons = []
    for row in accepted_large:
        medium_text = "".join(
            x["ja"] for x in medium
            if min(row["end"], x["end"]) - max(row["start"], x["start"]) > 0.1
        )
        comparisons.append(
            {
                "start": row["start"],
                "end": row["end"],
                "large_v3": row["ja"],
                "medium": medium_text,
                "similarity": round(similarity(row["ja"], medium_text), 6),
            }
        )

    (workdir / "large_review_raw.json").write_text(
        json.dumps(large_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (workdir / "model_comparison.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (workdir / "ja_final.json").write_text(
        json.dumps(filled, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"large_raw={len(large_rows)} large_accepted={len(accepted_large)} final={len(filled)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
