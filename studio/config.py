from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on", "cloud"}


RUNTIME_MODE = os.getenv("SUBTITLE_STUDIO_MODE", "").strip().lower()
CLOUD_MODE = _truthy(RUNTIME_MODE) if RUNTIME_MODE else os.name != "nt"
DATA_DIR = Path(
    os.getenv("SUBTITLE_STUDIO_DATA_DIR", str(ROOT / "studio_data"))
).expanduser().resolve()
ALLOW_LOCAL_OPEN = not CLOUD_MODE and _truthy(
    os.getenv("SUBTITLE_STUDIO_ALLOW_LOCAL_OPEN", "1")
)
SECURE_LOCAL_SECRETS = os.name == "nt" and not CLOUD_MODE
SECRET_POLICY = "windows_dpapi" if SECURE_LOCAL_SECRETS else "environment"
BASIC_AUTH_USERNAME = os.getenv("SUBTITLE_STUDIO_USERNAME", "").strip()
BASIC_AUTH_PASSWORD = os.getenv("SUBTITLE_STUDIO_PASSWORD", "")
BASIC_AUTH_ENABLED = bool(BASIC_AUTH_USERNAME and BASIC_AUTH_PASSWORD)


PROVIDER_KEY_ENV = {
    "asr": "SUBTITLE_ASR_API_KEY",
    "translator": "SUBTITLE_TRANSLATOR_API_KEY",
    "text_reviewer": "SUBTITLE_REVIEWER_API_KEY",
}
