"""Запросы к Gemini (REST через aiohttp) и подгонка ответа под сообщение Telegram."""
import asyncio
import re

import aiohttp

TG_LIMIT = 4096  # предел длины сообщения Telegram (в единицах UTF-16)
ASK_TIMEOUT = 30  # секунд на ответ одной модели
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"

SYSTEM_PROMPT = (
    "Тебя зовут Epsilon, ты ассистент в Telegram. Не представляйся и не называй своё имя сам — "
    "только если прямо спросят, как тебя зовут или кто ты. "
    "Никогда не здоровайся и не прощайся: не начинай ответ с «Здравствуйте», «Добрый день», "
    "«Приветствую» и подобного, даже если пользователь поздоровался или это первое сообщение — "
    "сразу переходи к сути. На одно «привет» без вопроса коротко спроси, чем помочь, без приветствия. "
    "Отвечай на языке вопроса, по делу и понятно, в официальном деловом стиле: обращайся на «вы», "
    "без сленга, шуток и фамильярности. "
    "Ответ — строго не длиннее 4096 символов: если тема большая, дай главное и предложи уточнить. "
    "Пиши обычным текстом без Markdown: без **, #, таблиц, блоков кода ``` и `инлайн-кода`; "
    "ссылки — просто адресом, без [текст](url); списки — строками с «•»."
)

# строка-ограждение блока кода: ``` или ```python, можно с пробелами в начале
FENCE = re.compile(r"\s*```[^`]*\Z")


class AIError(Exception):
    """Ни одна модель не ответила. Текст — ошибки по моделям, без ключа."""


class BlockedError(AIError):
    """Google заблокировал сам вопрос (promptFeedback.blockReason) — другие модели не спрашиваем."""


def tg_len(text):
    """Длина так, как считает Telegram: эмодзи и редкие символы — по 2 единицы."""
    return len(text.encode("utf-16-le")) // 2


def clean_line(line):
    """Чистит одну строку вне блока кода."""
    if re.fullmatch(r"\s*\*+\s*", line):  # строка из одних звёздочек: пустой ** или линия ***
        return ""
    line = re.sub(r"^#{1,6} +", "", line)
    line = re.sub(r"^( *)[*-] +", r"\1• ", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"\1 (\2)", line)
    # только парный жирный не внутри слова: 2**10, **kwargs, a**b не трогаем
    return re.sub(r"(?<!\w)\*\*(?=\S)(.+?)(?<=\S)\*\*(?!\w)", r"\1", line)


def clean(text):
    """Убирает Markdown, который Telegram без parse_mode показал бы звёздочками и решётками.
    Ограждения ``` удаляются, строки кода между ними не меняются."""
    # упрощение: чистим построчно — жирный через перенос строки не снимется; понадобится — склеивать куски вне кода
    lines, code = [], False
    for line in text.split("\n"):
        if FENCE.match(line):
            code = not code  # незакрытый блок — всё после ограждения остаётся кодом
            continue
        lines.append(line if code else clean_line(line))
    return "\n".join(lines).strip()


def fit(text, limit=TG_LIMIT):
    """Обрезает текст под одно сообщение: по абзацу, иначе по предложению, иначе по пробелу, в конце «…»."""
    if tg_len(text) <= limit:
        return text
    # режем по единицам UTF-16; половинку эмодзи на границе ignore выбросит
    cut = text.encode("utf-16-le")[:(limit - 1) * 2].decode("utf-16-le", "ignore")
    half = len(cut) // 2
    if (i := cut.rfind("\n")) > half:
        cut = cut[:i]
    elif (i := max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))) > half:
        cut = cut[:i + 1]
    elif m := re.search(r"\s\S{0,49}\Z", cut):
        # последний пробел не дальше 50 символов от конца — не рвём слово и флаг-эмодзи
        cut = cut[:m.start()]
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
            # error у Google бывает и строкой, а не объектом
            error = data.get("error") if isinstance(data, dict) else None
            message = (error.get("message") or "") if isinstance(error, dict) else ""
            errors.append(f"{model}: HTTP {status} {message}".rstrip())
            # модель отключена, не успела или перегружена — пробуем следующую
            if status in (404, 408, 429) or status >= 500:
                continue
            # 400/402/403 (кривой запрос, оплата, неверный ключ) у других моделей будет тем же
            raise AIError("; ".join(errors))
        text = extract_text(data)
        if text is None:
            feedback = data.get("promptFeedback") if isinstance(data, dict) else None
            reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
            if reason:
                # заблокирован сам вопрос — другие модели ответят так же
                raise BlockedError(f"{model}: blocked {reason}")
        text = clean(text) if text else ""
        if not text:
            errors.append(f"{model}: empty")
            continue
        return fit(text), model
    raise AIError("; ".join(errors) or "не задано ни одной модели")
