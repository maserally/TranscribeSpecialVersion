import json
import re
from pathlib import Path
from typing import Any

import httpx


WHISPER_MODEL_CHOICES = [
    "tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"
]


def _json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    return json.loads(text)


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 300):
        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.base_url = base
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(timeout=timeout, headers=self.headers)

    def list_models(self) -> list[str]:
        response = self.client.get(f"{self.base_url}/models")
        response.raise_for_status()
        data = response.json().get("data", [])
        return sorted({str(item["id"]) for item in data if item.get("id")})

    def chat_json(self, model: str, system: str, user: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": model,
            "temperature": 0.05,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        if response.status_code >= 400:
            payload.pop("response_format", None)
            response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
        return _json_from_text(str(content))

    def transcribe(self, model: str, wav_path: Path, language: str = "ja") -> dict[str, Any]:
        data = {
            "model": model,
            "language": language,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }
        with wav_path.open("rb") as file:
            response = self.client.post(
                f"{self.base_url}/audio/transcriptions",
                data=data,
                files={"file": (wav_path.name, file, "audio/wav")},
            )
        response.raise_for_status()
        return response.json()


class OllamaProvider:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def list_models(self) -> list[str]:
        response = self.client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return sorted({str(item["name"]) for item in response.json().get("models", [])})

    def chat_json(self, model: str, system: str, user: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.05, "num_ctx": 4096, "num_predict": 512},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
        }
        response = self.client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        return _json_from_text(response.json()["message"]["content"])


def cached_whisper_models(cache_dir: Path | None = None) -> list[str]:
    cache_dir = cache_dir or Path.home() / ".cache" / "whisper"
    models = [path.stem for path in cache_dir.glob("*.pt")] if cache_dir.exists() else []
    preferred = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
    return sorted(set(models), key=lambda x: (preferred.index(x) if x in preferred else 99, x))


def whisper_model_catalog(cache_dir: Path | None = None) -> list[dict[str, Any]]:
    installed = set(cached_whisper_models(cache_dir))
    choices = list(dict.fromkeys([*WHISPER_MODEL_CHOICES, *sorted(installed)]))
    return [{"id": model, "installed": model in installed} for model in choices]
