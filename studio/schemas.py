from typing import Literal

from pydantic import BaseModel, Field, field_validator


ProviderKind = Literal["local_whisper", "local_ollama", "openai_compatible"]
QualityProfile = Literal["precision", "balanced", "recall"]
SourceLanguage = Literal["ja", "ko"]


class ProviderSettings(BaseModel):
    kind: ProviderKind
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class JobOptions(BaseModel):
    input_path: str
    output_name: str = ""
    source_language: SourceLanguage = "ja"
    target_language: Literal["zh-CN"] = "zh-CN"
    profile: QualityProfile = "balanced"
    asr: ProviderSettings = Field(
        default_factory=lambda: ProviderSettings(kind="local_whisper", model="medium")
    )
    verifier_model: str = "large-v3"
    translator: ProviderSettings = Field(
        default_factory=lambda: ProviderSettings(
            kind="local_ollama", base_url="http://127.0.0.1:11434", model="qwen2.5:7b-instruct"
        )
    )
    remove_chinese_periods: bool = True
    publish_mode: bool = True
    create_soft_subtitle_video: bool = True
    create_hard_subtitle_video: bool = False
    enable_gap_recovery: bool = True

    @field_validator("input_path", mode="before")
    @classmethod
    def clean_input_path(cls, value):
        text = str(value or "").strip()
        quote_pairs = (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"))
        changed = True
        while changed and len(text) > 1:
            changed = False
            for left, right in quote_pairs:
                if text.startswith(left) and text.endswith(right):
                    text = text[len(left) : -len(right)].strip()
                    changed = True
        return text


class ModelListRequest(BaseModel):
    provider: ProviderSettings


class SavedProviderSettings(BaseModel):
    asr: ProviderSettings
    translator: ProviderSettings
    verifier_model: str = "large-v3"
