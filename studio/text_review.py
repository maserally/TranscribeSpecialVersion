from __future__ import annotations

from typing import Any, Callable

from .languages import language_info, source_text
from .schemas import ProviderSettings
from .translation import SOURCE_SCRIPT, audit_translation, provider_from_settings


BATCH_SIZE = 8
CONTEXT_SIZE = 3


SYSTEM_PROMPT = """你是影视字幕的最终文本校正员。输入含源语言识别文本、简体中文译文、前后文和已确认术语。
你的任务是修正高度确定的识别错字、漏字、中文误译、否定/动作方向错误、代词、人名与术语不一致，并让中文自然简洁。
必须忠实保留原意、语气、拒绝、疑问和成人语境，不得净化、扩写、猜测剧情或把不确定内容改成确定事实。
绝对不得合并、拆分、遗漏或调换目标字幕；每个目标 id 必须恰好返回一次。不要返回或修改时间戳。
若无法确定就保持原文和译文不变。source 必须保持输入的源语言，zh 必须是简体中文且不要使用句号。
只输出 JSON 对象：{"items":[{"id":整数,"source":"校正后的源文","zh":"校正后的中文","reason":"简短原因"}],"glossary":[{"source":"源文术语","zh":"统一中文"}]}。"""


def _validated_items(parsed: dict[str, Any], expected_ids: list[int]) -> list[dict[str, Any]]:
    items = parsed.get("items")
    if not isinstance(items, list) or len(items) != len(expected_ids):
        raise ValueError("校正模型返回的字幕数量不一致")
    by_id: dict[int, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("校正模型返回了无效字幕对象")
        try:
            item_id = int(item.get("id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("校正模型返回了无效字幕 ID") from exc
        if item_id in by_id:
            raise ValueError("校正模型返回了重复字幕 ID")
        by_id[item_id] = item
    if set(by_id) != set(expected_ids):
        raise ValueError("校正模型改变了字幕 ID")
    return [by_id[item_id] for item_id in expected_ids]


def _glossary_items(values: dict[str, str]) -> list[dict[str, str]]:
    return [{"source": source, "zh": zh} for source, zh in list(values.items())[-80:]]


def _update_glossary(values: dict[str, str], parsed: dict[str, Any]):
    items = parsed.get("glossary", [])
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        zh = str(item.get("zh", "")).strip().replace("。", "")
        if source and zh and len(source) <= 40 and len(zh) <= 40:
            values[source] = zh
    while len(values) > 80:
        values.pop(next(iter(values)))


def _too_long(candidate: str, original: str) -> bool:
    return len(candidate) > max(len(original) * 2.5, len(original) + 24)


def review_cues(
    rows: list[dict[str, Any]],
    settings: ProviderSettings,
    progress: Callable[[int, int, str], None] | None = None,
    *,
    source_language: str = "ja",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Review subtitle text in batches without changing cue IDs or timing."""
    provider = provider_from_settings(settings)
    language = language_info(source_language)
    originals = [dict(row) for row in rows]
    output = [dict(row) for row in rows]
    glossary: dict[str, str] = {}
    changes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejected_count = 0
    invalid_batches = 0

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch_end = min(len(rows), batch_start + BATCH_SIZE)
        expected_ids = list(range(batch_start + 1, batch_end + 1))
        context_start = max(0, batch_start - CONTEXT_SIZE)
        context_end = min(len(rows), batch_end + CONTEXT_SIZE)
        context = []
        for index in range(context_start, context_end):
            current = output[index] if index < batch_start else originals[index]
            context.append(
                {
                    "id": index + 1,
                    "source": source_text(current),
                    "zh": str(current.get("zh", "")),
                    "target": batch_start <= index < batch_end,
                }
            )
        request = {
            "source_language": language["name"],
            "confirmed_glossary": _glossary_items(glossary),
            "context": context,
            "targets": [item for item in context if item["target"]],
        }
        parsed = provider.chat_json(settings.model, SYSTEM_PROMPT, request)
        try:
            reviewed_items = _validated_items(parsed, expected_ids)
        except ValueError as first_error:
            request["format_error"] = str(first_error)
            parsed = provider.chat_json(
                settings.model,
                SYSTEM_PROMPT + "\n上次输出结构无效。只修复 JSON 结构，并严格保持 targets 的 ID 和数量。",
                request,
            )
            try:
                reviewed_items = _validated_items(parsed, expected_ids)
            except ValueError as second_error:
                invalid_batches += 1
                rejected_count += len(expected_ids)
                rejected.append(
                    {
                        "ids": expected_ids,
                        "reason": str(second_error),
                    }
                )
                if progress:
                    progress(batch_end, len(rows), f"校正批次结构无效，已保留原文：{expected_ids[0]}-{expected_ids[-1]}")
                continue

        _update_glossary(glossary, parsed)
        for item_id, item in zip(expected_ids, reviewed_items):
            index = item_id - 1
            original = originals[index]
            original_source = source_text(original).strip()
            original_zh = str(original.get("zh", "")).strip()
            candidate_source = str(item.get("source", "")).strip()
            candidate_zh = str(item.get("zh", "")).strip().replace("。", "")
            reason = str(item.get("reason", "")).strip()
            original_problems = set(audit_translation(original_source, original_zh, source_language))
            candidate_problems = set(audit_translation(candidate_source, candidate_zh, source_language))
            new_problems = sorted(candidate_problems - original_problems)
            reject_reason = ""
            if not candidate_source or not candidate_zh:
                reject_reason = "校正结果包含空文本"
            elif SOURCE_SCRIPT[source_language].search(original_source) and not SOURCE_SCRIPT[
                source_language
            ].search(candidate_source):
                reject_reason = f"校正结果不再包含{language['name']}文字"
            elif _too_long(candidate_source, original_source) or _too_long(candidate_zh, original_zh):
                reject_reason = "校正结果长度异常"
            elif new_problems:
                reject_reason = "校正引入新的语义审计问题：" + "；".join(new_problems)

            row = dict(original)
            row["id"] = item_id
            row["source"] = original_source
            row.pop("ja", None)
            row["zh"] = original_zh
            if reject_reason:
                rejected_count += 1
                warnings = list(row.get("text_review_warnings", []))
                warnings.append(reject_reason)
                row["text_review_warnings"] = warnings
                rejected.append({"id": item_id, "reason": reject_reason})
            else:
                row["source"] = candidate_source
                row["zh"] = candidate_zh
                if candidate_problems:
                    row["translation_warnings"] = sorted(candidate_problems)
                else:
                    row.pop("translation_warnings", None)
                if candidate_source != original_source or candidate_zh != original_zh:
                    row["text_review_changed"] = True
                    row["text_review_reason"] = reason or "上下文文本校正"
                    changes.append(
                        {
                            "id": item_id,
                            "before_source": original_source,
                            "after_source": candidate_source,
                            "before_zh": original_zh,
                            "after_zh": candidate_zh,
                            "reason": row["text_review_reason"],
                        }
                    )
            output[index] = row
        if progress:
            progress(batch_end, len(rows), f"最终文本校正 {batch_end}/{len(rows)}")

    audit = {
        "enabled": True,
        "provider": settings.kind,
        "model": settings.model,
        "cue_count": len(rows),
        "changed_count": len(changes),
        "rejected_count": rejected_count,
        "invalid_batches": invalid_batches,
        "glossary": _glossary_items(glossary),
        "changes": changes,
        "rejected": rejected,
    }
    return output, audit
