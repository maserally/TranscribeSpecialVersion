from typing import Literal

from pydantic import BaseModel, Field, field_validator


ProviderKind = Literal[
    "local_whisper", "accuracy_ensemble", "local_ollama", "openai_compatible"
]
QualityProfile = Literal["precision", "balanced", "recall"]
SourceLanguage = Literal["ja", "ko"]


class ProviderSettings(BaseModel):
    kind: ProviderKind
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class CloudWorkerSettings(BaseModel):
    enabled: bool = False
    host: str = ""
    port: int = Field(default=22, ge=1, le=65535)
    username: str = "root"
    password: str = ""
    private_key_path: str = ""
    remote_dir: str = "/root/subtitle-worker"
    model_dir: str = "/root/autodl-tmp/subtitle-models"
    auto_setup: bool = True


class JobOptions(BaseModel):
    input_path: str
    output_name: str = ""
    output_dir: str = ""
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
    text_reviewer: ProviderSettings = Field(
        default_factory=lambda: ProviderSettings(
            kind="local_ollama", base_url="http://127.0.0.1:11434", model="qwen2.5:7b-instruct"
        )
    )
    remove_chinese_periods: bool = True
    publish_mode: bool = True
    create_soft_subtitle_video: bool = True
    create_hard_subtitle_video: bool = False
    cloud_stage_only: bool = False

    @field_validator("input_path", "output_dir", mode="before")
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


class FolderScanRequest(BaseModel):
    input_dir: str
    recursive: bool = True

    @field_validator("input_dir", mode="before")
    @classmethod
    def clean_input_dir(cls, value):
        return JobOptions.clean_input_path(value)


class FolderPickerRequest(BaseModel):
    initial_dir: str = ""
    title: str = "选择文件夹"

    @field_validator("initial_dir", mode="before")
    @classmethod
    def clean_initial_dir(cls, value):
        return JobOptions.clean_input_path(value)


class FolderBatchRequest(BaseModel):
    input_dir: str
    output_dir: str = ""
    recursive: bool = True
    selected_files: list[str] | None = None
    options: JobOptions

    @field_validator("input_dir", "output_dir", mode="before")
    @classmethod
    def clean_directories(cls, value):
        return JobOptions.clean_input_path(value)


class ModelListRequest(BaseModel):
    provider: ProviderSettings
    role: Literal["asr", "translator", "text_reviewer"] | None = None


class SavedProviderSettings(BaseModel):
    asr: ProviderSettings
    translator: ProviderSettings
    text_reviewer: ProviderSettings = Field(
        default_factory=lambda: ProviderSettings(
            kind="local_ollama", base_url="http://127.0.0.1:11434", model="qwen2.5:7b-instruct"
        )
    )
    verifier_model: str = "large-v3"
    cloud_worker: CloudWorkerSettings = Field(default_factory=CloudWorkerSettings)


class CloudWorkerRequest(BaseModel):
    cloud_worker: CloudWorkerSettings


class JobOutputSettings(BaseModel):
    create_soft_subtitle_video: bool = False
    create_hard_subtitle_video: bool = False
