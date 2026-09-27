"""Общее для тестов: словари апдейтов Telegram, поддельный Telegram (MockedSession),
поддельный Gemini (FakeGemini) и поддельный ask (FakeAsk)."""
import asyncio
import copy
import itertools
from datetime import datetime, timezone

from aiogram.client.session.base import BaseSession
from aiogram.exceptions import ClientDecodeError, TelegramBadRequest, TelegramRetryAfter
from aiogram.methods import AnswerCallbackQuery, DeleteMessage, DeleteMessages, EditMessageText, SendMessage
from aiogram.types import Chat, Message

# id с запасом по длине, чтобы "есть ли id в тексте" не совпадало случайно
USER = 111111
USER2 = 333333
OTHER = 222222
OTHER2 = 444444
ADMIN = 999999
START = [{"type": "bot_command", "offset": 0, "length": 6}]
PHOTO = [{"file_id": "x", "file_unique_id": "y", "width": 1, "height": 1}]
STICKER = {"file_id": "s", "file_unique_id": "t", "type": "regular", "width": 1, "height": 1,
           "is_animated": False, "is_video": False}
# создатель бота для меню
NAME = "Автор"
URL = "https://t.me/author"

# id входящих сообщений: 1, 2, … (отправленные ботом — от 100, не пересекаются)
_ids = itertools.count(1)


def reset_ids():
    """Начать id входящих заново с 1 — вызывать перед каждым тестом, чтобы не дойти до 100."""
    global _ids
    _ids = itertools.count(1)


def update(sender, text=None, **fields):
    """Словарь обновления Telegram: сообщение от sender в его личный чат (id чата = id пользователя).
    Id сообщения — новый при каждом вызове: update(...)["message"]["message_id"]."""
    message = {
        "message_id": next(_ids),
        "date": 0,
        "chat": {"id": sender, "type": "private"},
        "from": {"id": sender, "is_bot": False, "first_name": "T"},
        **fields,
    }
    if text is not None:
        message["text"] = text
    return {"update_id": 1, "message": message}


def start(sender):
    """Словарь команды /start от sender."""
    return update(sender, "/start", entities=START)


def callback(sender, data, message_id):
    """Словарь нажатия инлайн-кнопки с data под сообщением бота message_id в личном чате sender."""
    return {"update_id": 1, "callback_query": {
        "id": "cb",
        "from": {"id": sender, "is_bot": False, "first_name": "T"},
        "chat_instance": "ci",
        "data": data,
        # date не 0: с date 0 aiogram считает сообщение недоступным (InaccessibleMessage)
        "message": {
            "message_id": message_id,
            "date": 1,
            "chat": {"id": sender, "type": "private"},
            "from": {"id": 42, "is_bot": True, "first_name": "Epsilon"},
            "text": "После очистки сессия не сохранится",
        },
    }}


def rows(markup):
    """Надписи нижних кнопок (ReplyKeyboardMarkup) по рядам."""
    return [[b.text for b in row] for row in markup.keyboard]


class MockedSession(BaseSession):
    """Поддельный Telegram: запоминает все запросы бота, в сеть не ходит.

    Отправленным сообщениям даёт id по порядку: 100, 101, …; пары (SendMessage, id) — в ids.
    fail=True — любой запрос падает с TelegramBadRequest; fail_delete=True — падают только
    DeleteMessage и DeleteMessages (запрос при этом всё равно запоминается).
    decode_error=True — любой запрос падает с ClientDecodeError (кривой ответ, это не TelegramAPIError);
    fail_send=True — всегда падает SendMessage; fail_answer=True — падает AnswerCallbackQuery;
    fail_edit=True — падает EditMessageText;
    retry_after=N — столько следующих SendMessage ответят TelegramRetryAfter(retry_after=0).
    """

    def __init__(self):
        super().__init__()
        self.requests, self.ids, self.next_id = [], [], 100
        self.fail, self.fail_delete = False, False
        self.decode_error, self.fail_send, self.fail_answer, self.retry_after = False, False, False, 0
        self.fail_edit = False

    async def make_request(self, bot, method, timeout=None):
        # отдаём управление, как настоящая сеть: параллельные задачи перемешиваются
        await asyncio.sleep(0)
        self.requests.append(method)
        if self.fail:
            raise TelegramBadRequest(method, "Bad Request: chat not found")
        if self.decode_error:
            raise ClientDecodeError("Failed to deserialize object", ValueError("кривой JSON"), "<html>")
        if self.fail_delete and isinstance(method, (DeleteMessage, DeleteMessages)):
            raise TelegramBadRequest(method, "Bad Request: message can't be deleted for everyone")
        if self.fail_answer and isinstance(method, AnswerCallbackQuery):
            raise TelegramBadRequest(method, "Bad Request: query is too old")
        if self.fail_edit and isinstance(method, EditMessageText):
            raise TelegramBadRequest(method, "Bad Request: message to edit not found")
        if isinstance(method, SendMessage):
            if self.retry_after:
                self.retry_after -= 1
                raise TelegramRetryAfter(method, "Too Many Requests: retry after 0", retry_after=0)
            if self.fail_send:
                raise TelegramBadRequest(method, "Bad Request: chat not found")
            self.ids.append((method, self.next_id))
            self.next_id += 1
            return Message(message_id=self.next_id - 1, date=datetime.now(timezone.utc),
                           chat=Chat(id=method.chat_id, type="private"), text=method.text).as_(bot)
        return True

    async def stream_content(self, url, headers=None, timeout=30, chunk_size=65536, raise_for_status=True):
        raise NotImplementedError

    async def close(self):
        pass


class FakeAsk:
    """Вместо ai.ask для бота: запоминает (history, question), отвечает ("ответ N", "m1"),
    где N — номер вызова. error — бросить это исключение вместо ответа; answer — отвечать этим текстом;
    gate — asyncio.Event: «Gemini думает», пока тест не вызовет gate.set() (вызов в calls уже записан)."""

    def __init__(self):
        self.calls, self.error, self.answer, self.gate = [], None, None, None

    async def __call__(self, history, question):
        # копия: бот может потом менять тот же список; deepcopy сохраняет тип (list остаётся list)
        self.calls.append((copy.deepcopy(history), question))
        number = len(self.calls)
        # отдаём управление, как настоящий запрос в сеть
        await asyncio.sleep(0)
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.answer or f"ответ {number}", "m1"


def gemini_ok(text):
    """Ответ Gemini с одной текстовой частью."""
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


class FakeResponse:
    def __init__(self, status, data=None):
        self.status, self.data = status, data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self, *args, **kwargs):
        return self.data


class Failing:
    """Ответ, который при входе бросает исключение (таймаут, обрыв связи)."""

    def __init__(self, error):
        self.error = error

    async def __aenter__(self):
        raise self.error

    async def __aexit__(self, *exc):
        return False


class FakeGemini:
    """Вместо aiohttp-сессии для Gemini: {модель: (код, JSON) или исключение}.
    Все запросы — в calls: (url, json, headers)."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json, headers))
        # .../models/{model}:generateContent -> model
        answer = self.routes[url.rsplit("/", 1)[1].split(":")[0]]
        return Failing(answer) if isinstance(answer, BaseException) else FakeResponse(*answer)

    def models(self):
        """Имена моделей, к которым обращались, по порядку."""
        return [url.rsplit("/", 1)[1].split(":")[0] for url, _, _ in self.calls]
