import argparse
import json
import subprocess
from collections import deque
from pathlib import Path

import webrtcvad


SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2


def collect_vad(media_path: str, aggressiveness: int = 2):
    vad = webrtcvad.Vad(aggressiveness)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", media_path,
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "s16le", "-acodec", "pcm_s16le", "-",
        ],
        stdout=subprocess.PIPE,
    )
    ring = deque(maxlen=10)  # 300 ms look-back
    active = False
    start = 0.0
    silence_run = 0
    segments = []
    frame_index = 0

    while True:
        frame = proc.stdout.read(FRAME_BYTES)
        if len(frame) < FRAME_BYTES:
            break
        now = frame_index * FRAME_MS / 1000
        voiced = bool(vad.is_speech(frame, SAMPLE_RATE))
        frame_index += 1

        if not active:
            ring.append((now, voiced))
            if len(ring) == ring.maxlen and sum(v for _, v in ring) >= 7:
                active = True
                start = ring[0][0]
                silence_run = 0
                ring.clear()
        else:
            if voiced:
                silence_run = 0
            else:
                silence_run += 1
            if silence_run >= 17:  # 510 ms without voice
                end = now - (silence_run - 1) * FRAME_MS / 1000
                if end - start >= 0.24:
                    segments.append([max(0.0, start - 0.18), end + 0.22])
                active = False
                ring.clear()
                silence_run = 0

    if active:
        end = frame_index * FRAME_MS / 1000
        if end - start >= 0.24:
            segments.append([max(0.0, start - 0.18), end])
    proc.wait()
    if proc.returncode:
        raise RuntimeError(f"ffmpeg exited with {proc.returncode}")
    return merge_and_split(segments)


def merge_and_split(segments):
    merged = []
    for start, end in segments:
        if merged and start - merged[-1][1] <= 0.45 and end - merged[-1][0] <= 28:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    split = []
    for start, end in merged:
        cursor = start
        while end - cursor > 28:
            split.append([cursor, cursor + 28])
            cursor += 27.5
        if end - cursor >= 0.24:
            split.append([cursor, end])
    return [{"start": round(s, 3), "end": round(e, 3)} for s, e in split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", type=int, default=2, choices=[0, 1, 2, 3])
    args = ap.parse_args()
    segments = collect_vad(args.media, args.mode)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    total = sum(x["end"] - x["start"] for x in segments)
    print(f"segments={len(segments)} voiced_seconds={total:.1f} voiced_minutes={total/60:.2f}")


if __name__ == "__main__":
    main()
