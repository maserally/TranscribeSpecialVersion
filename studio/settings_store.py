from __future__ import annotations

import base64
import ctypes
import json
import os
from ctypes import wintypes
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "studio_data" / "settings" / "provider_settings.json"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(data: bytes):
    buffer = ctypes.create_string_buffer(data)
    blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    return blob, buffer


def _protect(value: str) -> str:
    if not value:
        return ""
    raw = value.encode("utf-8")
    if os.name != "nt":
        return "plain:" + base64.b64encode(raw).decode("ascii")
    source, source_buffer = _blob(raw)
    target = _DataBlob()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(source),
        "SubtitleCreate API Key",
        None,
        None,
        None,
        0,
        ctypes.byref(target),
    ):
        raise OSError("Windows DPAPI 无法加密 API Key")
    try:
        encrypted = ctypes.string_at(target.pbData, target.cbData)
        return "dpapi:" + base64.b64encode(encrypted).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(target.pbData)
        del source_buffer


def _unprotect(value: str) -> str:
    if not value:
        return ""
    if value.startswith("plain:"):
        return base64.b64decode(value[6:]).decode("utf-8")
    if not value.startswith("dpapi:") or os.name != "nt":
        return ""
    encrypted = base64.b64decode(value[6:])
    source, source_buffer = _blob(encrypted)
    target = _DataBlob()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None, 0, ctypes.byref(target)
    ):
        return ""
    try:
        raw = ctypes.string_at(target.pbData, target.cbData)
        return raw.decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(target.pbData)
        del source_buffer


def load_provider_settings() -> dict[str, Any]:
    defaults = {
        "asr": {"kind": "local_whisper", "base_url": "https://api.openai.com/v1", "api_key": "", "model": "medium"},
        "translator": {"kind": "local_ollama", "base_url": "http://127.0.0.1:11434", "api_key": "", "model": "qwen2.5:7b-instruct"},
        "text_reviewer": {"kind": "local_ollama", "base_url": "http://127.0.0.1:11434", "api_key": "", "model": "qwen2.5:7b-instruct"},
        "verifier_model": "large-v3",
    }
    if not SETTINGS_PATH.exists():
        return defaults
    try:
        stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        for name in ("asr", "translator", "text_reviewer"):
            section = stored.get(name, {})
            defaults[name].update(
                {key: str(section.get(key, defaults[name][key])) for key in ("kind", "base_url", "model")}
            )
            defaults[name]["api_key"] = _unprotect(str(section.get("api_key", "")))
        defaults["verifier_model"] = str(stored.get("verifier_model", defaults["verifier_model"]))
    except Exception:
        pass
    return defaults


def save_provider_settings(settings: dict[str, Any]) -> Path:
    stored: dict[str, Any] = {"version": 2}
    for name in ("asr", "translator", "text_reviewer"):
        section = settings.get(name, {})
        stored[name] = {
            "kind": str(section.get("kind", "")),
            "base_url": str(section.get("base_url", "")).strip(),
            "model": str(section.get("model", "")).strip(),
            "api_key": _protect(str(section.get("api_key", ""))),
        }
    stored["verifier_model"] = str(settings.get("verifier_model", "")).strip()
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(SETTINGS_PATH)
    return SETTINGS_PATH
