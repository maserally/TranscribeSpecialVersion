from __future__ import annotations

import base64
import ctypes
import json
import os
from ctypes import wintypes
from typing import Any

from .config import DATA_DIR, PROVIDER_KEY_ENV, SECURE_LOCAL_SECRETS


SETTINGS_PATH = DATA_DIR / "settings" / "provider_settings.json"


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
    if not SECURE_LOCAL_SECRETS:
        return ""
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
    if not value.startswith("dpapi:") or not SECURE_LOCAL_SECRETS:
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


def load_provider_settings(*, expose_secrets: bool | None = None) -> dict[str, Any]:
    defaults = {
        "asr": {"kind": "local_whisper", "base_url": "https://api.openai.com/v1", "api_key": "", "model": "medium"},
        "translator": {"kind": "local_ollama", "base_url": "http://127.0.0.1:11434", "api_key": "", "model": "qwen2.5:7b-instruct"},
        "text_reviewer": {"kind": "local_ollama", "base_url": "http://127.0.0.1:11434", "api_key": "", "model": "qwen2.5:7b-instruct"},
        "verifier_model": "large-v3",
        "cloud_worker": {
            "enabled": False,
            "host": "",
            "port": 22,
            "username": "root",
            "password": "",
            "private_key_path": "",
            "remote_dir": "/root/subtitle-worker",
            "auto_setup": True,
        },
    }
    if SETTINGS_PATH.exists():
        try:
            stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            for name in ("asr", "translator", "text_reviewer"):
                section = stored.get(name, {})
                defaults[name].update(
                    {
                        key: str(section.get(key, defaults[name][key]))
                        for key in ("kind", "base_url", "model")
                    }
                )
                defaults[name]["api_key"] = _unprotect(str(section.get("api_key", "")))
            defaults["verifier_model"] = str(
                stored.get("verifier_model", defaults["verifier_model"])
            )
            worker = stored.get("cloud_worker", {})
            for key in (
                "enabled", "host", "port", "username", "private_key_path",
                "remote_dir", "auto_setup",
            ):
                if key in worker:
                    defaults["cloud_worker"][key] = worker[key]
            defaults["cloud_worker"]["password"] = _unprotect(
                str(worker.get("password", ""))
            )
        except Exception:
            pass
    expose = SECURE_LOCAL_SECRETS if expose_secrets is None else expose_secrets
    for name, env_name in PROVIDER_KEY_ENV.items():
        env_value = os.getenv(env_name, "")
        saved_value = defaults[name]["api_key"]
        defaults[name]["api_key_configured"] = bool(env_value or saved_value)
        if expose:
            defaults[name]["api_key"] = saved_value
        else:
            defaults[name]["api_key"] = ""
    worker_password = str(defaults["cloud_worker"].get("password", ""))
    defaults["cloud_worker"]["password_configured"] = bool(worker_password)
    if not expose:
        defaults["cloud_worker"]["password"] = ""
    return defaults


def save_provider_settings(settings: dict[str, Any]) -> Path:
    existing: dict[str, Any] = {}
    if SETTINGS_PATH.exists():
        try:
            existing = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    stored: dict[str, Any] = {"version": 2}
    for name in ("asr", "translator", "text_reviewer"):
        section = settings.get(name, {})
        incoming_key = str(section.get("api_key", ""))
        existing_key = str(existing.get(name, {}).get("api_key", ""))
        stored[name] = {
            "kind": str(section.get("kind", "")),
            "base_url": str(section.get("base_url", "")).strip(),
            "model": str(section.get("model", "")).strip(),
            # The browser intentionally leaves an already-saved secret blank.
            # Blank therefore means "keep", not "erase".
            "api_key": _protect(incoming_key) if incoming_key else existing_key,
        }
    stored["verifier_model"] = str(settings.get("verifier_model", "")).strip()
    worker = settings.get("cloud_worker", {})
    stored["cloud_worker"] = {
        "enabled": bool(worker.get("enabled", False)),
        "host": str(worker.get("host", "")).strip(),
        "port": int(worker.get("port", 22)),
        "username": str(worker.get("username", "root")).strip(),
        "password": (
            _protect(str(worker.get("password", "")))
            if str(worker.get("password", ""))
            else str(existing.get("cloud_worker", {}).get("password", ""))
        ),
        "private_key_path": str(worker.get("private_key_path", "")).strip(),
        "remote_dir": str(worker.get("remote_dir", "/root/subtitle-worker")).strip(),
        "auto_setup": bool(worker.get("auto_setup", True)),
    }
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(SETTINGS_PATH)
    return SETTINGS_PATH


def resolve_provider_api_keys(settings):
    """Fill blank job provider keys from environment or encrypted local settings."""
    resolved = settings.model_copy(deep=True)
    saved = load_provider_settings(expose_secrets=True)
    for name, env_name in PROVIDER_KEY_ENV.items():
        provider = getattr(resolved, name)
        if not provider.api_key:
            provider.api_key = os.getenv(env_name, "") or str(
                saved.get(name, {}).get("api_key", "")
            )
    return resolved
