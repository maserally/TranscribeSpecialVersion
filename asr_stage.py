import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import whisper


HALLUCINATIONS = (
    "ご視聴ありがとうございました",
    "チャンネル登録",
    "字幕をご覧いただき",
    "最後までご覧",
    "おやすみなさい",
)
NONLEXICAL_RE = re.compile(r"^[ぁあぃいうぅえぇえぉおんっッはぁー〜～・]+$")
PUNCT_ONLY_RE = re.compile(r"^[\s、。！？!?…・〜～ー]+$")


def select_windows(events, threshold=0.15, nonlexical_factor=1.2):
    selected = [
        x for x in events
        if x["speech_score"] >= threshold
        and x["speech_score"] > x["nonlexical_score"] * nonlexical_factor
    ]
    merged = []
    for row in selected:
        start, end = row["start"], row["end"]
        if merged and start <= merged[-1]["end"] + 0.28 and end - merged[-1]["start"] <= 25:
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["speech_scores"].append(row["speech_score"])
            merged[-1]["nonlexical_scores"].append(row["nonlexical_score"])
        else:
            merged.append(
                {
                    "start": max(0.0, start - 0.18),
                    "end": end + 0.22,
                    "speech_scores": [row["speech_score"]],
                    "nonlexical_scores": [row["nonlexical_score"]],
                }
            )
    return merged


def pack_windows(windows, max_audio=26.0, separator=0.55):
    packs = []
    current = None
    for window in windows:
        duration = window["end"] - window["start"]
        needed = duration + (separator if current and current["mappings"] else 0)
        if current and current["duration"] + needed > max_audio:
            packs.append(current)
            current = None
        if current is None:
            current = {"duration": 0.0, "mappings": []}
        if current["mappings"]:
            current["duration"] += separator
        packed_start = current["duration"]
        current["mappings"].append(
            {
                "packed_start": packed_start,
                "packed_end": packed_start + duration,
                "original_start": window["start"],
                "original_end": window["end"],
                "speech_score": max(window["speech_scores"]),
                "nonlexical_score": max(window["nonlexical_scores"]),
            }
        )
        current["duration"] += duration
    if current and current["mappings"]:
        packs.append(current)
    return packs


def map_word_to_original(word, mappings):
    midpoint = (word["start"] + word["end"]) / 2
    mapping = next(
        (
            m for m in mappings
            if m["packed_start"] - 0.04 <= midpoint <= m["packed_end"] + 0.04
        ),
        None,
    )
    if mapping is None:
        return None
    mapped = dict(word)
    mapped["start"] = mapping["original_start"] + max(
        0.0, word["start"] - mapping["packed_start"]
    )
    mapped["end"] = mapping["original_start"] + min(
        mapping["original_end"] - mapping["original_start"],
        word["end"] - mapping["packed_start"],
    )
    mapped["start"] = max(mapping["original_start"], mapped["start"])
    mapped["end"] = min(mapping["original_end"], mapped["end"])
    mapped["event_speech_score"] = mapping["speech_score"]
    mapped["event_nonlexical_score"] = mapping["nonlexical_score"]
    if mapped["end"] <= mapped["start"]:
        return None
    return mapped


def repetition_score(text):
    chars = re.sub(r"[\s、。！？!?…・〜～ー]", "", text)
    if len(chars) < 6:
        return 0.0
    for size in range(1, min(10, len(chars) // 2) + 1):
        unit = chars[:size]
        repetitions = 0
        pos = 0
        while chars.startswith(unit, pos):
            repetitions += 1
            pos += size
        if repetitions >= 3:
            return pos / len(chars)
    return 0.0


def rejection_reasons(seg):
    text = seg["text"].strip()
    compact = re.sub(r"[\s、。！？!?…・〜～ー]", "", text)
    reasons = []
    if not compact or PUNCT_ONLY_RE.fullmatch(text):
        reasons.append("empty")
    if any(x in text for x in HALLUCINATIONS):
        reasons.append("known_hallucination")
    if seg["compression_ratio"] > 2.4:
        reasons.append("high_compression")
    if seg["avg_logprob"] < -0.75:
        reasons.append("low_logprob")
    if seg["no_speech_prob"] > 0.72:
        reasons.append("no_speech")
    if NONLEXICAL_RE.fullmatch(compact or "") and len(compact) <= 30:
        reasons.append("nonlexical_vocalization")
    if repetition_score(text) >= 0.65:
        reasons.append("repetition")
    return reasons


def dedupe_words(words):
    output = []
    for word in sorted(words, key=lambda x: (x["start"], x["end"])):
        token = word["word"].strip()
        if not token:
            continue
        if output:
            prev = output[-1]
            overlap = min(prev["end"], word["end"]) - max(prev["start"], word["start"])
            if token == prev["word"].strip() and overlap > -0.08:
                if word.get("probability", 0) > prev.get("probability", 0):
                    output[-1] = word
                continue
        output.append(word)
    return output


def build_sentences(words):
    sentences = []
    current = []

    def flush():
        nonlocal current
        if not current:
            return
        text = "".join(w["word"].strip() for w in current).strip()
        text = re.sub(r"([。！？!?])\1+", r"\1", text)
        if text and not PUNCT_ONLY_RE.fullmatch(text):
            sentences.append(
                {
                    "start": round(max(0, current[0]["start"] - 0.08), 3),
                    "end": round(current[-1]["end"] + 0.14, 3),
                    "ja": text,
                    "mean_word_probability": round(
                        sum(w.get("probability", 0.5) for w in current) / len(current), 6
                    ),
                }
            )
        current = []

    for word in words:
        if current:
            gap = word["start"] - current[-1]["end"]
            duration = current[-1]["end"] - current[0]["start"]
            chars = sum(len(w["word"].strip()) for w in current)
            if gap > 0.82 or duration >= 7.0 or chars >= 32:
                flush()
        current.append(word)
        token = word["word"].strip()
        duration = current[-1]["end"] - current[0]["start"]
        if re.search(r"[。！？!?]$", token) and duration >= 0.75:
            flush()
    flush()
    return sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--events", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--speech-threshold", type=float, default=0.15)
    ap.add_argument("--nonlexical-factor", type=float, default=1.2)
    args = ap.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    windows = select_windows(events, args.speech_threshold, args.nonlexical_factor)
    packs = pack_windows(windows)
    (workdir / "asr_windows.json").write_text(
        json.dumps(windows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"ASR windows={len(windows)} packed_batches={len(packs)} "
        f"duration={sum(x['end']-x['start'] for x in windows)/60:.2f} min",
        flush=True,
    )
    audio = whisper.load_audio(args.media)
    model = whisper.load_model(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    candidates = []
    accepted_words = []

    for index, pack in enumerate(packs, 1):
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
            word_timestamps=True,
            hallucination_silence_threshold=1.0,
            verbose=False,
        )
        for seg in result["segments"]:
            absolute = dict(seg)
            absolute["packed_batch"] = index
            absolute["event_speech_score"] = max(
                m["speech_score"] for m in pack["mappings"]
            )
            absolute["event_nonlexical_score"] = max(
                m["nonlexical_score"] for m in pack["mappings"]
            )
            absolute["rejection_reasons"] = rejection_reasons(absolute)
            absolute_words = []
            for word in seg.get("words", []):
                mapped = map_word_to_original(word, pack["mappings"])
                if mapped:
                    absolute_words.append(mapped)
            if absolute_words:
                absolute["start"] = min(w["start"] for w in absolute_words)
                absolute["end"] = max(w["end"] for w in absolute_words)
            absolute["words"] = absolute_words
            candidates.append(absolute)
            if not absolute["rejection_reasons"]:
                accepted_words.extend(absolute_words)
        if index % 5 == 0 or index == len(packs):
            print(f"Transcribed {index}/{len(packs)}", flush=True)
            (workdir / "asr_candidates.partial.json").write_text(
                json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    candidates_path = workdir / "asr_candidates.json"
    candidates_path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    words = dedupe_words(accepted_words)
    sentences = build_sentences(words)
    (workdir / "ja_sentences.json").write_text(
        json.dumps(sentences, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"candidates={len(candidates)} accepted={sum(not x['rejection_reasons'] for x in candidates)} "
        f"sentences={len(sentences)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
