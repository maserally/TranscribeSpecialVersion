from __future__ import annotations

import re
from typing import Any, Callable

from .providers import OllamaProvider, OpenAICompatibleProvider
from .languages import language_info, source_text
from .schemas import ProviderSettings


NEGATIVE_ZH = re.compile(r"不|没|別|别|未|无|不是|并非|不要|不同|错|住手|停下")
NEGATIVE_SOURCE = {
    "ja": re.compile(r"ない|ません|じゃない|なく|てない|聞いてない|違う"),
    "ko": re.compile(r"않|못|없|아니|아닙|안|말[아어]|싫|틀렸|하지\s*마"),
}
LEXICAL_NAI_JA = re.compile(r"危ない|汚い|少ない|幼い|切ない|もったいない|つまらない")
SOURCE_SCRIPT = {
    "ja": re.compile(r"[ぁ-ゟ゠-ヿ]"),
    "ko": re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]"),
}
TRANSLATION_BATCH_SIZE = 12
TRANSLATION_CONTEXT_SIZE = 8
PLAN_CHUNK_SIZE = 120


PLAN_SYSTEM_PROMPT = """你是影视字幕的全片语境分析员。输入是一段按时间排列的源语言字幕和之前累计的全片信息。
只建立后续翻译所需的语境索引：概括本段场景、人物关系、称谓、反复出现的名字/术语、说话语气和需要跨句理解的梗。
不得编造画面或对白，不确定信息必须标记为不确定。成人影片内容不得回避或净化。
只输出 JSON：{"summary":"本段摘要","characters":["人物/关系/称谓"],"glossary":[{"source":"源文","zh":"建议统一中文"}]}。"""


SYSTEM_PROMPTS = {
    "ja": """你是专业日中影视字幕译者。翻译 targets 中的日语；context 和 global_context 只供理解，绝不能把相邻句内容并入错误的 id。
输出自然、简洁、忠实的简体中文字幕；保持否定、拒绝、疑问、人称和语气，不得补写原文没有的信息。
结合整批对白、全片场景摘要、人物关系、术语表和前后文判断省略成分、动作方向和人名；人名音译成中文，不得残留日文假名。动作的进入、取出、脱落方向不得反译。
不得把 やめて 译成“别停”或“继续”；默认译为“住手、停下、不要这样”。
聞いてない 必须保留否定，待って 必须包含“等一下”，勘弁 必须保留求饶含义。
成人语境也必须忠实翻译，不得把拒绝反译成同意。
输入可能来自成人影片：不得回避、净化或用含糊词替换明确内容，但也不得把普通对白擅自粗俗化、色情化。
译文必须像中国观众会说的话，优先传达语境功能，不做逐词字典直译；双关、俗语、儿歌、数数歌和文化梗应按上下文意译。
喘息、呻吟和背景音乐不得被扩写成不存在的对白。若 target 的 asr_review 显示两种识别明显不一致，且原文本身不通顺，宁可输出“【听不清】”并标记 source_uncertain，也不得编造合理剧情。
原文不完整时保留省略感；无法确认的标签必须保留不确定性。
必须按输入顺序返回每个 target id，不能遗漏、合并或增加。中文不要使用句号。
只输出 JSON：{"items":[{"id":整数,"zh":"译文","naturalness_risk":布尔值,"source_uncertain":布尔值,"note":"简短说明"}],"continuity":{"scene_summary":"更新后的当前场景记忆","characters":["人物/关系/称谓"]},"glossary":[{"source":"源文术语","zh":"统一中文"}]}。""",
    "ko": """你是专业韩中影视字幕译者。翻译 targets 中的韩语；context 和 global_context 只供理解，绝不能把相邻句内容并入错误的 id。
输出自然、简洁、忠实的简体中文字幕；保持否定、拒绝、疑问、人称、敬语层级和语气，不得补写原文没有的信息。
结合整批对白、全片场景摘要、人物关系、术语表和前后文判断省略成分、动作方向和人名；人名音译成中文，不得残留韩文字符。动作的进入、取出、脱落方向不得反译。
不得把 하지 마 或 그만해 译成“继续”或同意；应保留“不要、住手、停下”的含义。
기다려 必须保留“等一下”，안 들었어／못 들었어 必须保留“没听到”，봐줘／살려줘 必须保留求饶或求救含义。
成人语境也必须忠实翻译，不得把拒绝反译成同意。
输入可能来自成人影片：不得回避、净化或用含糊词替换明确内容，但也不得把普通对白擅自粗俗化、色情化。
译文必须像中国观众会说的话，优先传达语境功能，不做逐词字典直译；双关、俗语、儿歌、数数歌和文化梗应按上下文意译。
喘息、呻吟和背景音乐不得被扩写成不存在的对白。若 target 的 asr_review 显示两种识别明显不一致，且原文本身不通顺，宁可输出“【听不清】”并标记 source_uncertain，也不得编造合理剧情。
原文不完整时保留省略感；无法确认的标签必须保留不确定性。
必须按输入顺序返回每个 target id，不能遗漏、合并或增加。中文不要使用句号。
只输出 JSON：{"items":[{"id":整数,"zh":"译文","naturalness_risk":布尔值,"source_uncertain":布尔值,"note":"简短说明"}],"continuity":{"scene_summary":"更新后的当前场景记忆","characters":["人物/关系/称谓"]},"glossary":[{"source":"源文术语","zh":"统一中文"}]}。""",
}


def provider_from_settings(settings: ProviderSettings):
    if settings.kind == "local_ollama":
        return OllamaProvider(settings.base_url or "http://127.0.0.1:11434")
    if settings.kind == "openai_compatible":
        return OpenAICompatibleProvider(settings.base_url, settings.api_key)
    raise ValueError(f"Unsupported translation provider: {settings.kind}")


def audit_translation(source: str, zh: str, source_language: str = "ja") -> list[str]:
    problems = []
    source_name = language_info(source_language)["name"]
    if not zh.strip():
        problems.append("译文为空")
    if "。" in zh or zh.rstrip().endswith("."):
        problems.append("中文不得包含句号")
    negative_re = NEGATIVE_SOURCE[source_language]
    negative_source = LEXICAL_NAI_JA.sub("", source) if source_language == "ja" else source
    if negative_re.search(negative_source) and not NEGATIVE_ZH.search(zh):
        problems.append(f"{source_name}的否定或纠正含义没有保留")
    if source_language == "ja" and "拝む" in source and re.search(r"朝拜|膜拜", zh):
        problems.append("拝む 在当前人物语境中被生硬直译，应结合上下文译成自然的夸赞或想见面")
    if (
        source_language == "ja"
        and "一本でも" in source
        and re.search(r"ごぼう|ニンジン|人参", source)
        and re.search(r"即使.*也是|哪怕.*也是", zh)
    ):
        problems.append("疑似数数歌或语言梗被逐词直译，中文语义不成立")
    if re.search(r"やめて|やめろ", source) and ("别停" in zh or "继续" in zh):
        problems.append("やめて 被反译")
    if re.search(r"やめて|やめろ", source) and not re.search(r"住手|停下|停一|不要|别这样", zh):
        problems.append("やめて 缺少停止或拒绝含义")
    if "待って" in source and "等" not in zh:
        problems.append("待って 必须包含等一下")
    if "聞いてない" in source and not re.search(r"没听|不知道|没听说|不清楚", zh):
        problems.append("聞いてない 必须保留没听或不知道")
    if "勘弁" in source and not re.search(r"饶|放过|受不了|别再", zh):
        problems.append("勘弁 的求饶语气未体现")
    if "抜けちゃった" in source and not re.search(r"掉|脱|出来|滑|拔|抜", zh):
        problems.append("抜けちゃった 的脱落方向没有保留")
    if re.search(r"入れます|入れる", source) and not re.search(r"放|进|插|装|加入", zh):
        problems.append("入れます 的放入方向没有保留")
    if re.search(r"하지\s*마|그만해", source) and not re.search(r"不要|别|住手|停下|够了", zh):
        problems.append("韩语制止或拒绝含义没有保留")
    if "기다려" in source and "等" not in zh:
        problems.append("기다려 必须包含等一下")
    if re.search(r"안\s*들었|못\s*들었", source) and not re.search(r"没听|没听到|不知道", zh):
        problems.append("韩语未听到的否定含义没有保留")
    if re.search(r"봐\s*줘|살려\s*줘", source) and not re.search(r"放过|饶|救|帮", zh):
        problems.append("韩语求饶或求救含义没有保留")
    if SOURCE_SCRIPT[source_language].search(zh):
        problems.append(f"中文残留{source_name}字符")
    return problems


def safe_high_risk(source: str, zh: str, source_language: str = "ja") -> str:
    if re.search(r"やめて|やめろ", source):
        if re.search(r"住手|停下|停一|不要|别这样", zh) and "继续" not in zh and "别停" not in zh:
            return zh
        if "本当に" in source:
            return "真的，住手"
        if "ちょっと" in source:
            return "请先停一下"
        return "住手"
    if "聞いてない" in source:
        return "没听说过吗？" if "?" in source or "？" in source else "我没听说"
    if "勘弁してください" in source:
        return "请放过我吧"
    if "勘弁" in source:
        return "饶了我吧"
    if "抜けちゃった" in source:
        return zh if re.search(r"掉|脱|出来|滑|拔", zh) else "啊，掉出来了"
    if re.search(r"入れます|入れる", source):
        return zh if re.search(r"放|进|插|装|加入", zh) else "要放进去了"
    if source_language == "ko":
        if re.search(r"하지\s*마|그만해", source):
            return "住手"
        if "기다려" in source:
            return "等一下"
        if re.search(r"안\s*들었|못\s*들었", source):
            return "我没听到"
        if re.search(r"살려\s*줘", source):
            return "救救我"
        if re.search(r"봐\s*줘", source):
            return "放过我吧"
    return zh


def _validated_translation_items(
    parsed: dict[str, Any], expected_ids: list[int]
) -> list[dict[str, Any]]:
    items = parsed.get("items")
    if not isinstance(items, list) or len(items) != len(expected_ids):
        raise ValueError("翻译模型返回的字幕数量不一致")
    by_id: dict[int, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("翻译模型返回了无效字幕对象")
        try:
            item_id = int(item.get("id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("翻译模型返回了无效字幕 ID") from exc
        if item_id in by_id:
            raise ValueError("翻译模型返回了重复字幕 ID")
        by_id[item_id] = item
    if set(by_id) != set(expected_ids):
        raise ValueError("翻译模型改变了字幕 ID")
    return [by_id[item_id] for item_id in expected_ids]


def _merge_unique(current: list[str], values: Any, limit: int = 60) -> list[str]:
    if isinstance(values, list):
        for value in values:
            text = str(value).strip()
            if text and text not in current:
                current.append(text[:200])
    return current[-limit:]


def _update_glossary(glossary: dict[str, str], parsed: dict[str, Any]):
    values = parsed.get("glossary", [])
    if not isinstance(values, list):
        return
    for item in values:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        zh = str(item.get("zh", "")).strip().replace("。", "")
        if source and zh and len(source) <= 60 and len(zh) <= 60:
            glossary[source] = zh
    while len(glossary) > 100:
        glossary.pop(next(iter(glossary)))


def build_translation_plan(
    rows: list[dict[str, Any]],
    settings: ProviderSettings,
    progress: Callable[[int, int, str], None] | None = None,
    *,
    source_language: str = "ja",
) -> dict[str, Any]:
    provider = provider_from_settings(settings)
    language = language_info(source_language)
    scenes: list[dict[str, Any]] = []
    characters: list[str] = []
    glossary: dict[str, str] = {}
    for start in range(0, len(rows), PLAN_CHUNK_SIZE):
        end = min(len(rows), start + PLAN_CHUNK_SIZE)
        request = {
            "source_language": language["name"],
            "previous_global_context": {
                "recent_scenes": scenes[-4:],
                "characters": characters[-40:],
                "glossary": [
                    {"source": source, "zh": zh}
                    for source, zh in list(glossary.items())[-80:]
                ],
            },
            "segment": [
                {"id": index + 1, "source": source_text(rows[index])}
                for index in range(start, end)
            ],
        }
        parsed = provider.chat_json(settings.model, PLAN_SYSTEM_PROMPT, request)
        summary = str(parsed.get("summary", "")).strip()[:1600]
        scenes.append({"start_id": start + 1, "end_id": end, "summary": summary})
        _merge_unique(characters, parsed.get("characters"))
        _update_glossary(glossary, parsed)
        if progress:
            progress(end, len(rows), f"全片语境索引 {end}/{len(rows)}")
    return {
        "version": 1,
        "source_language": source_language,
        "cue_count": len(rows),
        "scenes": scenes,
        "characters": characters,
        "glossary": [
            {"source": source, "zh": zh} for source, zh in glossary.items()
        ],
    }


def _plan_glossary(plan: dict[str, Any] | None) -> dict[str, str]:
    glossary: dict[str, str] = {}
    for item in (plan or {}).get("glossary", []):
        if isinstance(item, dict):
            source = str(item.get("source", "")).strip()
            zh = str(item.get("zh", "")).strip()
            if source and zh:
                glossary[source] = zh
    return glossary


def _request_translation_items(
    provider,
    model: str,
    system_prompt: str,
    request: dict[str, Any],
    expected_ids: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parsed = provider.chat_json(model, system_prompt, request)
    try:
        return _validated_translation_items(parsed, expected_ids), [parsed]
    except ValueError as first_error:
        retry_request = {**request, "format_error": str(first_error)}
        parsed = provider.chat_json(
            model,
            system_prompt
            + "\n上次输出结构无效。只修复 JSON 结构，严格保持 targets 的 ID、顺序和数量。",
            retry_request,
        )
        try:
            return _validated_translation_items(parsed, expected_ids), [parsed]
        except ValueError:
            if len(expected_ids) == 1:
                raise
    midpoint = len(expected_ids) // 2
    left_ids, right_ids = expected_ids[:midpoint], expected_ids[midpoint:]
    targets = request["targets"]
    left_request = {**request, "targets": targets[:midpoint]}
    right_request = {**request, "targets": targets[midpoint:]}
    left_items, left_responses = _request_translation_items(
        provider, model, system_prompt, left_request, left_ids
    )
    right_items, right_responses = _request_translation_items(
        provider, model, system_prompt, right_request, right_ids
    )
    return left_items + right_items, left_responses + right_responses


def translate_cues(
    rows: list[dict[str, Any]],
    settings: ProviderSettings,
    progress: Callable[[int, int, str], None] | None = None,
    *,
    source_language: str = "ja",
    target_language: str = "zh-CN",
    existing: list[dict[str, Any]] | None = None,
    checkpoint: Callable[[list[dict[str, Any]]], None] | None = None,
    translation_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if target_language != "zh-CN":
        raise ValueError(f"不支持的目标语言：{target_language}")
    provider = provider_from_settings(settings)
    system_prompt = SYSTEM_PROMPTS[source_language]
    output: list[dict[str, Any]] = []
    for index, saved in enumerate(existing or []):
        if index >= len(rows) or source_text(saved) != source_text(rows[index]):
            break
        output.append(dict(saved))

    glossary = _plan_glossary(translation_plan)
    continuity: dict[str, Any] = {
        "scene_summary": "",
        "characters": list((translation_plan or {}).get("characters", []))[-40:],
    }
    for batch_start in range(len(output), len(rows), TRANSLATION_BATCH_SIZE):
        batch_end = min(len(rows), batch_start + TRANSLATION_BATCH_SIZE)
        expected_ids = list(range(batch_start + 1, batch_end + 1))
        context_start = max(0, batch_start - TRANSLATION_CONTEXT_SIZE)
        context_end = min(len(rows), batch_end + TRANSLATION_CONTEXT_SIZE)
        context = []
        for index in range(context_start, context_end):
            item = {"id": index + 1, "source": source_text(rows[index])}
            if index < len(output):
                item["zh"] = output[index].get("zh", "")
            item["target"] = batch_start <= index < batch_end
            context.append(item)
        targets = []
        for index in range(batch_start, batch_end):
            target = {"id": index + 1, "source": source_text(rows[index])}
            if rows[index].get("asr_review"):
                target["asr_review"] = rows[index]["asr_review"]
            targets.append(target)
        request = {
            "global_context": {
                "film_plan": translation_plan or {},
                "rolling_continuity": continuity,
                "confirmed_glossary": [
                    {"source": source, "zh": zh}
                    for source, zh in list(glossary.items())[-100:]
                ],
            },
            "context": context,
            "targets": targets,
        }
        translated_items, translation_responses = _request_translation_items(
            provider, settings.model, system_prompt, request, expected_ids
        )
        for response in translation_responses:
            _update_glossary(glossary, response)
        parsed_continuity = translation_responses[-1].get("continuity", {})
        if isinstance(parsed_continuity, dict):
            summary = str(parsed_continuity.get("scene_summary", "")).strip()
            if summary:
                continuity["scene_summary"] = summary[:2000]
            continuity["characters"] = _merge_unique(
                list(continuity.get("characters", [])),
                parsed_continuity.get("characters"),
                60,
            )

        items_by_id = {item_id: item for item_id, item in zip(expected_ids, translated_items)}
        problems_by_id: dict[int, list[str]] = {}
        for item_id in expected_ids:
            item = items_by_id[item_id]
            original = source_text(rows[item_id - 1])
            zh = str(item.get("zh", "")).strip().replace("。", "")
            problems = audit_translation(original, zh, source_language)
            if item.get("naturalness_risk"):
                problems.append("模型标记译文存在生硬直译或语境自然度风险")
            if item.get("source_uncertain"):
                problems.append("模型标记源语识别不确定，禁止用猜测补全对白")
            if problems:
                problems_by_id[item_id] = problems

        if problems_by_id:
            repair_targets = [
                {
                    **next(target for target in targets if target["id"] == item_id),
                    "previous_zh": str(items_by_id[item_id].get("zh", "")),
                    "problems": problems,
                }
                for item_id, problems in problems_by_id.items()
            ]
            repair_request = {**request, "targets": repair_targets}
            try:
                repaired_items, repair_responses = _request_translation_items(
                    provider,
                    settings.model,
                    system_prompt
                    + "\n以下译文未通过自动语义审计。只集中修复 targets 中列出的条目并保持 ID。",
                    repair_request,
                    list(problems_by_id),
                )
                items_by_id.update(
                    {item_id: item for item_id, item in zip(problems_by_id, repaired_items)}
                )
                for response in repair_responses:
                    _update_glossary(glossary, response)
            except ValueError:
                pass

        residue_ids = [
            item_id
            for item_id in expected_ids
            if SOURCE_SCRIPT[source_language].search(
                str(items_by_id[item_id].get("zh", ""))
            )
        ]
        if residue_ids:
            residue_request = {
                **request,
                "targets": [
                    {
                        **next(target for target in targets if target["id"] == item_id),
                        "previous_zh": str(items_by_id[item_id].get("zh", "")),
                    }
                    for item_id in residue_ids
                ],
            }
            try:
                repaired_items, _ = _request_translation_items(
                    provider,
                    settings.model,
                    system_prompt
                    + "\n只修复 targets 译文中残留的源语言字符；人名音译，其余忠实转成简体中文。",
                    residue_request,
                    residue_ids,
                )
                items_by_id.update(
                    {item_id: item for item_id, item in zip(residue_ids, repaired_items)}
                )
            except ValueError:
                pass

        for item_id in expected_ids:
            source_row = rows[item_id - 1]
            row = dict(source_row)
            original = source_text(row)
            item = items_by_id[item_id]
            zh = str(item.get("zh", "")).strip().replace("。", "")
            zh = safe_high_risk(original, zh, source_language)
            remaining = audit_translation(original, zh, source_language)
            if item.get("naturalness_risk"):
                remaining.append("模型仍标记译文自然度风险")
            if item.get("source_uncertain"):
                remaining.append("双模型识别存在歧义，建议人工听校")
            if remaining:
                row["translation_warnings"] = remaining
            row["id"] = item_id
            row["source"] = original
            row.pop("ja", None)
            row["zh"] = zh
            output.append(row)
        if checkpoint:
            checkpoint(output)
        if progress:
            progress(batch_end, len(rows), f"语境批量翻译 {batch_end}/{len(rows)}")
    return output
