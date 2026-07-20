from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import whisper

from ensemble_common import choose_consensus, normalize_transcript


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("media")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", choices=("ja", "ko"), required=True)
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    conflict_indices = [index for index, row in enumerate(rows) if row.get("needs_third_vote")]
    model = None
    audio = None
    if conflict_indices:
        audio = whisper.load_audio(args.media)
        model = whisper.load_model(args.model, device="cuda" if torch.cuda.is_available() else "cpu")

    audit = []
    for number, row in enumerate(rows, 1):
        qwen_text = str(row.get("qwen_source") or "").strip()
        cohere_text = str(row.get("cohere_source") or "").strip()
        whisper_text = ""
        if row.get("needs_third_vote"):
            clip = audio[int(row["start"] * 16000) : int(row["end"] * 16000)]
            result = model.transcribe(
                clip,
                language=args.language,
                task="transcribe",
                fp16=torch.cuda.is_available(),
                temperature=0,
                condition_on_previous_text=False,
                verbose=False,
            )
            whisper_text = str(result.get("text") or "").strip()
            print(f"large-v3 conflict {number}/{len(rows)}", flush=True)

        if whisper_text:
            final_text, winner, similarities = choose_consensus(
                qwen_text, cohere_text, whisper_text
            )
        elif normalize_transcript(qwen_text):
            final_text, winner = qwen_text, "qwen"
            similarities = {
                "qwen_cohere": float(row.get("qwen_cohere_similarity", 0)),
                "qwen_whisper": 0.0,
                "cohere_whisper": 0.0,
            }
        else:
            final_text, winner = cohere_text, "cohere"
            similarities = {
                "qwen_cohere": float(row.get("qwen_cohere_similarity", 0)),
                "qwen_whisper": 0.0,
                "cohere_whisper": 0.0,
            }
        row["whisper_source"] = whisper_text
        row["final_source"] = final_text
        row["ensemble_winner"] = winner
        row["similarities"] = similarities
        audit.append(
            {
                "start": row["start"],
                "end": row["end"],
                "review_reasons": row.get("review_reasons", []),
                "qwen": qwen_text,
                "cohere": cohere_text,
                "whisper": whisper_text,
                "winner": winner,
                "similarities": similarities,
            }
        )

    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.audit).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"large-v3 conflicts={len(conflict_indices)}", flush=True)


if __name__ == "__main__":
    main()
