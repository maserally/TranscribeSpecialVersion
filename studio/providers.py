import json
import re
from pathlib import Path
from typing import Any

import httpx


WHISPER_MODEL_CHOICES = [
    "tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"
]


def _json_from_text(text: str) -> dict[str, Any]:
    """Read the first JSON object from common OpenAI-compatible responses."""
    text = text.strip()
    candidates = [text]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.I)
    )
    decoder = json.JSONDecoder()
    errors: list[Exception] = []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(exc)
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate, index)
            except json.JSONDecodeError as exc:
                errors.append(exc)
                continue
            if isinstance(value, dict):
                return value
    detail = str(errors[-1]) if errors else "响应中没有 JSON 对象"
    raise ValueError(f"模型返回内容无法解析为 JSON 对象：{detail}")


def normalize_openai_base_url(base_url: str) -> str:
    """Accept a host, /v1 base, or a complete OpenAI-compatible endpoint."""
    base = base_url.strip().rstrip("/")
    for suffix in ("/audio/transcriptions", "/chat/completions", "/models"):
        if base.endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
            break
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def supports_segment_timestamps(model: str) -> bool:
    return not (model.startswith("gpt-4o") and "transcribe" in model)


def _raise_provider_error(response: httpx.Response):
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = re.sub(r"\s+", " ", response.text).strip()[:1200]
        suffix = f"；服务响应：{detail}" if detail else ""
        raise RuntimeError(
            f"远程服务返回 HTTP {response.status_code}（{response.request.url}）{suffix}"
        ) from exc


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 300):
        self.base_url = normalize_openai_base_url(base_url)
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(timeout=timeout, headers=self.headers)

    def list_models(self) -> list[str]:
        response = self.client.get(f"{self.base_url}/models")
        _raise_provider_error(response)
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
        parse_error: ValueError | None = None
        for attempt in range(3):
            if attempt:
                payload["messages"] = [
                    {"role": "system", "content": system},
                    {
                        "role": "system",
                        "content": (
                            "The previous response was invalid JSON. Return exactly one complete "
                            "JSON object with double-quoted keys and values; do not omit commas."
                        ),
                    },
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ]
            response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
            if response.status_code >= 400 and "response_format" in payload:
                payload.pop("response_format", None)
                response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
            _raise_provider_error(response)
            content = response.json()["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    str(x.get("text", "")) for x in content if isinstance(x, dict)
                )
            try:
                return _json_from_text(str(content))
            except ValueError as exc:
                parse_error = exc
        raise RuntimeError("远程模型连续 3 次返回无效 JSON，已停止当前句以避免错位") from parse_error

    def transcribe(self, model: str, wav_path: Path, language: str = "ja") -> dict[str, Any]:
        data = {
            "model": model,
            "language": language,
        }
        if supports_segment_timestamps(model):
            data["response_format"] = "verbose_json"
            data["timestamp_granularities[]"] = "segment"
        else:
            data["response_format"] = "json"
        with wav_path.open("rb") as file:
            response = self.client.post(
                f"{self.base_url}/audio/transcriptions",
                data=data,
                files={"file": (wav_path.name, file, "audio/wav")},
            )
        _raise_provider_error(response)
        return response.json()


class OllamaProvider:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def list_models(self) -> list[str]:
        response = self.client.get(f"{self.base_url}/api/tags")
        _raise_provider_error(response)
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
        parse_error: ValueError | None = None
        for attempt in range(3):
            if attempt:
                payload["messages"] = [
                    {"role": "system", "content": system},
                    {
                        "role": "system",
                        "content": "上次响应不是完整 JSON。只返回一个完整 JSON 对象，键和值使用双引号，不要漏逗号。",
                    },
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ]
            response = self.client.post(f"{self.base_url}/api/chat", json=payload)
            _raise_provider_error(response)
            try:
                return _json_from_text(response.json()["message"]["content"])
            except ValueError as exc:
                parse_error = exc
        raise RuntimeError("本地模型连续 3 次返回无效 JSON，已停止当前句以避免错位") from parse_error


def cached_whisper_models(cache_dir: Path | None = None) -> list[str]:
    cache_dir = cache_dir or Path.home() / ".cache" / "whisper"
    models = [path.stem for path in cache_dir.glob("*.pt")] if cache_dir.exists() else []
    preferred = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
    return sorted(set(models), key=lambda x: (preferred.index(x) if x in preferred else 99, x))


def whisper_model_catalog(cache_dir: Path | None = None) -> list[dict[str, Any]]:
    installed = set(cached_whisper_models(cache_dir))
    choices = list(dict.fromkeys([*WHISPER_MODEL_CHOICES, *sorted(installed)]))
    return [{"id": model, "installed": model in installed} for model in choices]
