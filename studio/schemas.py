from typing import Literal

from pydantic import BaseModel, Field


ProviderKind = Literal["local_whisper", "local_ollama", "openai_compatible"]
QualityProfile = Literal["precision", "balanced", "recall"]


class ProviderSettings(BaseModel):
    kind: ProviderKind
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class JobOptions(BaseModel):
    input_path: str
    output_name: str = ""
    source_language: str = "ja"
    target_language: str = "zh-CN"
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


class ModelListRequest(BaseModel):
    provider: ProviderSettings
