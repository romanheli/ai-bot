"""Запросы к Gemini (REST через aiohttp) и подгонка ответа под сообщение Telegram."""
import asyncio
import re

import aiohttp

TG_LIMIT = 4096  # предел длины сообщения Telegram (в единицах UTF-16)
ASK_TIMEOUT = 60  # секунд на ответ одной модели
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
RETRY_STATUSES = {404, 429, 500, 503}  # модель отключена или перегружена — пробуем следующую
URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"

SYSTEM_PROMPT = (
    "Ты полезный ассистент в Telegram. Отвечай на языке вопроса, по делу и понятно. "
    "Ответ — не длиннее 3500 символов: если тема большая, дай главное и предложи уточнить. "
    "Пиши обычным текстом без Markdown: без **, #, таблиц; списки — строками с «•»."
)


class AIError(Exception):
    """Ни одна модель не ответила. Текст — ошибки по моделям, без ключа."""


def tg_len(text):
    """Длина так, как считает Telegram: эмодзи и редкие символы — по 2 единицы."""
    return len(text.encode("utf-16-le")) // 2


def clean(text):
    """Убирает Markdown, который Telegram без parse_mode показал бы звёздочками и решётками."""
    text = text.replace("**", "")
    text = re.sub(r"^#{1,6} +", "", text, flags=re.M)
    text = re.sub(r"^( *)[*-] +", r"\1• ", text, flags=re.M)
    return text.strip()


def fit(text, limit=TG_LIMIT):
    """Обрезает текст под одно сообщение: по абзацу, иначе по предложению, в конце «…»."""
    if tg_len(text) <= limit:
        return text
    # режем по единицам UTF-16; половинку эмодзи на границе ignore выбросит
    cut = text.encode("utf-16-le")[:(limit - 1) * 2].decode("utf-16-le", "ignore")
    half = len(cut) // 2
    if (i := cut.rfind("\n")) > half:
        cut = cut[:i]
    elif (i := cut.rfind(". ")) > half:
        cut = cut[:i + 1]
    return cut.rstrip() + "…"


def build_body(history, question):
    """Тело запроса: системная инструкция + прошлые пары (вопрос, ответ) + новый вопрос."""
    contents = []
    for q, a in history:
        contents.append({"role": "user", "parts": [{"text": q}]})
        contents.append({"role": "model", "parts": [{"text": a}]})
    contents.append({"role": "user", "parts": [{"text": question}]})
    return {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]}, "contents": contents}


def extract_text(data):
    """Текст ответа без «мыслей» модели; нет текста (фильтр, пустой ответ) — None."""
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return text or None


async def ask(session, key, models, history, question):
    """Спрашивает модели по очереди, пока одна не ответит. -> (ответ под Telegram, модель)."""
    body = build_body(history, question)
    errors = []
    for model in models:
        try:
            # ключ в заголовке, а не в адресе — чтобы не попал в логи при ошибке
            async with session.post(URL.format(model), json=body, headers={"x-goog-api-key": key},
                                    timeout=aiohttp.ClientTimeout(total=ASK_TIMEOUT)) as r:
                status = r.status
                try:
                    data = await r.json(content_type=None)
                except ValueError:
                    data = None
        except asyncio.TimeoutError:
            errors.append(f"{model}: timeout")
            continue
        except aiohttp.ClientError:
            errors.append(f"{model}: connection")
            continue
        if status != 200:
            message = data.get("error", {}).get("message", "") if isinstance(data, dict) else ""
            errors.append(f"{model}: HTTP {status} {message}".rstrip())
            if status in RETRY_STATUSES:
                continue
            # 400/403 (кривой запрос, неверный ключ) у других моделей будет тем же
            raise AIError("; ".join(errors))
        text = extract_text(data)
        if text is None:
            errors.append(f"{model}: empty")
            continue
        return fit(clean(text)), model
    raise AIError("; ".join(errors) or "не задано ни одной модели")
