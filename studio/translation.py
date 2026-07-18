from __future__ import annotations

import re
from typing import Any, Callable

from .providers import OllamaProvider, OpenAICompatibleProvider
from .schemas import ProviderSettings


NEGATIVE_JA = re.compile(r"ない|ません|じゃない|なく|てない|聞いてない|違う")
NEGATIVE_ZH = re.compile(r"不|没|別|别|未|无|不是|并非|不要|不同|错")


SYSTEM_PROMPT = """你是专业日中影视字幕译者。只翻译 target 中的日语，context 仅供理解，绝不能把相邻句译进 target。
输出自然、简洁、忠实的简体中文字幕；保持否定、拒绝、疑问、人称和语气，不得补写原文没有的信息。
不得把 やめて 译成“别停”或“继续”；默认译为“住手、停下、不要这样”。
聞いてない 必须保留否定，待って 必须包含“等一下”，勘弁 必须保留求饶含义。
成人语境也必须忠实翻译，不得把拒绝反译成同意。
原文不完整时保留省略感；无法确认的标签必须保留不确定性。
中文不要使用句号。只输出 JSON：{"id":整数,"zh":"译文"}。"""


def provider_from_settings(settings: ProviderSettings):
    if settings.kind == "local_ollama":
        return OllamaProvider(settings.base_url or "http://127.0.0.1:11434")
    if settings.kind == "openai_compatible":
        return OpenAICompatibleProvider(settings.base_url, settings.api_key)
    raise ValueError(f"Unsupported translation provider: {settings.kind}")


def audit_translation(ja: str, zh: str) -> list[str]:
    problems = []
    if not zh.strip():
        problems.append("译文为空")
    if "。" in zh or zh.rstrip().endswith("."):
        problems.append("中文不得包含句号")
    if NEGATIVE_JA.search(ja) and not NEGATIVE_ZH.search(zh):
        problems.append("日文的否定或纠正含义没有保留")
    if "やめて" in ja and ("别停" in zh or "继续" in zh):
        problems.append("やめて 被反译")
    if "やめて" in ja and not re.search(r"住手|停下|停一|不要|别这样", zh):
        problems.append("やめて 缺少停止或拒绝含义")
    if "待って" in ja and "等" not in zh:
        problems.append("待って 必须包含等一下")
    if "聞いてない" in ja and not re.search(r"没听|不知道|没听说|不清楚", zh):
        problems.append("聞いてない 必须保留没听或不知道")
    if "勘弁" in ja and not re.search(r"饶|放过|受不了|别再", zh):
        problems.append("勘弁 的求饶语气未体现")
    if re.search(r"[ぁ-ゟ゠-ヿ]", zh):
        problems.append("中文残留日文假名")
    return problems


def safe_high_risk(ja: str, zh: str) -> str:
    if "やめて" in ja:
        if "本当に" in ja:
            return "真的，住手"
        if "ちょっと" in ja:
            return "请先停一下"
        return "住手"
    if "聞いてない" in ja:
        return "没听说过吗？" if "?" in ja or "？" in ja else "我没听说"
    if "勘弁してください" in ja:
        return "请放过我吧"
    if "勘弁" in ja:
        return "饶了我吧"
    return zh


def translate_cues(
    rows: list[dict[str, Any]],
    settings: ProviderSettings,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    provider = provider_from_settings(settings)
    output = []
    for index, source in enumerate(rows):
        row = dict(source)
        context = [
            {"id": i + 1, "ja": rows[i].get("ja", "")}
            for i in range(max(0, index - 2), min(len(rows), index + 3))
        ]
        request = {"context": context, "target": {"id": index + 1, "ja": row.get("ja", "")}}
        parsed = provider.chat_json(settings.model, SYSTEM_PROMPT, request)
        zh = str(parsed.get("zh", "")).strip().replace("。", "")
        problems = audit_translation(row.get("ja", ""), zh)
        if problems:
            parsed = provider.chat_json(
                settings.model,
                SYSTEM_PROMPT + "\n上次译文未通过审计，请修正：" + "；".join(problems),
                request,
            )
            zh = str(parsed.get("zh", "")).strip().replace("。", "")
        zh = safe_high_risk(row.get("ja", ""), zh)
        remaining = audit_translation(row.get("ja", ""), zh)
        if remaining:
            row["translation_warnings"] = remaining
        row["id"] = index + 1
        row["zh"] = zh
        output.append(row)
        if progress:
            progress(index + 1, len(rows), zh)
    return output

