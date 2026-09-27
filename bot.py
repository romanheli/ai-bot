"""Telegram-бот: отвечает на вопросы через Gemini, копии вопросов шлёт владельцу через лог-бота."""
import asyncio
import contextlib
import functools
import html
import logging
import os
import re
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import AiogramError, TelegramNetworkError, TelegramRetryAfter
from aiogram.filters import CommandStart
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup

import ai

CLEAR_BUTTON = "Очистить сессию"
MENU_BUTTON = "Главное меню"
MAIN_KB = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text=CLEAR_BUTTON), KeyboardButton(text=MENU_BUTTON)]],
                              resize_keyboard=True, is_persistent=True)
CONFIRM_TEXT = "После очистки сессия не сохранится"
ACCEPT_BUTTON = "Принять"
CANCEL_BUTTON = "Отменить"
CONFIRM_KB = InlineKeyboardMarkup(inline_keyboard=[[
    InlineKeyboardButton(text=ACCEPT_BUTTON, callback_data="clear:yes"),
    InlineKeyboardButton(text=CANCEL_BUTTON, callback_data="clear:no"),
]])
GIFT_BUTTON = "отправить интимку автору"
HISTORY_SIZE = 5  # сколько последних пар (вопрос, ответ) помнит бот
DENIED_TEXT = "Нет доступа."
ERROR_TEXT = "Нейросеть сейчас не отвечает, попробуй чуть позже."
CLEARED_TEXT = "Сессия очищена."
EMPTY_TEXT = "Сессия уже пуста."
ONLY_TEXT = "Пока понимаю только текст."
BUSY_TEXT = "Подожди, ещё отвечаю на прошлый вопрос."
BLOCKED_TEXT = "Нейросеть отказалась отвечать на этот вопрос — попробуй переформулировать."
LOG_ANSWER = 300  # сколько символов ответа показывать в логе
DELETE_BATCH = 100  # больше id за раз delete_messages не принимает
DELETE_AGE = 47 * 3600  # удаляем только сообщения моложе: старше 48 ч Telegram всё равно не удалит
DENIED_EVERY = 3600  # чужому отвечаем «Нет доступа» и пишем владельцу не чаще раза в час
STOPPED_TEXT = "Бот остановлен: остановка или перезагрузка сервера."
CRASH_NOTE = "Прошлый запуск оборвался без остановки: процесс убили или сервер упал."
MSK = timezone(timedelta(hours=3))  # в Москве нет перехода на летнее время
TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]+")  # токен бота в адресе запроса Telegram
now = time.monotonic  # часы для DELETE_AGE и DENIED_EVERY; тесты подменяют bot.now


def redact(text):
    """Вырезает токены ботов из текста (адреса запросов в ошибках aiogram)."""
    return TOKEN_RE.sub("bot<TOKEN>", text)


class RedactingFormatter(logging.Formatter):
    """Формат логов без токенов — и в сообщении, и в traceback."""

    def format(self, record):
        return redact(super().format(record))


def menu_text(creator_name, creator_url):
    """Текст главного меню (HTML)."""
    return ("Привет, я Epsilon, ИИ бот помощник\nТы можешь спросить меня о чем угодно\n\n"
            f"<i>создатель - <a href=\"{html.escape(creator_url, quote=True)}\">{html.escape(creator_name)}</a></i>")


def menu_kb(creator_url):
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=GIFT_BUTTON, url=creator_url)]])


def load_env(path):
    """Читает строки KEY=VALUE из файла в os.environ, уже заданное непустое не трогает."""
    if not os.path.exists(path):
        return
    # utf-8-sig: Блокнот ставит в начало файла BOM, иначе первый ключ был бы "﻿KEY"
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key.startswith("export "):  # строка в стиле shell: export KEY=VALUE
                key = key[len("export "):].strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].strip()  # KEY=VALUE # комментарий
            if not os.environ.get(key):
                os.environ[key] = value


def parse_ids(value):
    """"1, 2 3" -> {1, 2, 3}; пусто -> пустое множество."""
    ids = set()
    for part in re.split(r"[,\s]+", (value or "").strip()):
        if not part:
            continue
        if not part.isdigit():
            raise SystemExit(f"Telegram id должен быть числом, а не «{part}»")
        ids.add(int(part))
    return ids


def read_settings(env):
    """Настройки из переменных окружения (env — словарь, в main это os.environ); ошибка — SystemExit."""
    names = ("BOT_TOKEN", "LOG_BOT_TOKEN", "ADMIN_ID", "GEMINI_API_KEY", "CREATOR_USERNAME")
    missing = [name for name in names if not env.get(name)]
    if missing:
        raise SystemExit("Не заданы в .env: " + ", ".join(missing))
    admin = parse_ids(env["ADMIN_ID"])
    if len(admin) != 1:
        raise SystemExit("ADMIN_ID — ровно один Telegram id")
    username = env["CREATOR_USERNAME"].lstrip("@")
    return {
        "token": env["BOT_TOKEN"],
        "log_token": env["LOG_BOT_TOKEN"],
        "admin_id": admin.pop(),
        "allowed": parse_ids(env.get("ALLOWED_IDS")),
        "api_key": env["GEMINI_API_KEY"],
        "models": [m.strip() for m in env.get("GEMINI_MODELS", "").split(",") if m.strip()] or ai.DEFAULT_MODELS,
        "creator_name": env.get("CREATOR_NAME") or "@" + username,
        "creator_url": f"https://t.me/{username}",
    }


def who(user):
    """Подпись пользователя для лога: имя, @username, id."""
    name = f"@{user.username}, " if user.username else ""
    return f"{user.full_name} ({name}id {user.id})"


def short(text, size=LOG_ANSWER):
    return text if len(text) <= size else text[:size].rstrip() + "…"


@contextlib.asynccontextmanager
async def typing(bot, chat_id):
    """«печатает…» вверху чата, пока ждём Gemini. Telegram держит статус ~5 с — обновляем каждые 4."""
    async def keep():
        while True:
            await asyncio.sleep(4)
            with contextlib.suppress(AiogramError):
                await bot.send_chat_action(chat_id, "typing")

    with contextlib.suppress(AiogramError):
        await bot.send_chat_action(chat_id, "typing")
    task = asyncio.create_task(keep())
    try:
        yield
    finally:
        task.cancel()


def next_weekly_reset(now):
    """Ближайший сброс недельного лимита Claude (воскресенье 00:00 UTC) строго после now."""
    midnight = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(days=(6 - midnight.weekday()) % 7 or 7)


def fmt_msk(moment):
    """Время по Москве: 27.09.2026 03:00."""
    return moment.astimezone(MSK).strftime("%d.%m.%Y %H:%M")


def weekly_text(reset):
    return f"Недельный лимит Claude сбросился: {fmt_msk(reset)} МСК"


async def weekly_loop(log_bot, admin_id):
    """Раз в неделю лог-бот сообщает о сбросе лимита Claude."""
    # упрощение: если сервер лежал в момент сброса, сообщение за эту неделю не придёт
    while True:
        try:
            now = datetime.now(timezone.utc)
            reset = next_weekly_reset(now)
            # +5 с — проснуться точно после сброса, иначе при ранней побудке пришло бы дважды
            await asyncio.sleep((reset - now).total_seconds() + 5)
            if datetime.now(timezone.utc) < reset:  # проснулись раньше — ждём ещё
                continue
            await tell(log_bot, admin_id, weekly_text(reset))
        except Exception:
            # задача в фоне: без этого упала бы молча и напоминаний больше не было бы
            logging.exception("Недельное напоминание не сработало")


def banner(models, now):
    """Заметная ярко-зелёная рамка «бот запущен» для консоли хостинга."""
    line = "=" * 60
    return (f"\033[1;92m{line}\n"
            f"  EPSILON ЗАПУЩЕН  {fmt_msk(now)} МСК\n"
            f"  модели: {', '.join(models)}\n"
            f"{line}\033[0m")


async def tell(log_bot, admin_id, text):
    """Сообщение владельцу через лог-бота; не дошло — только предупреждение в консоль."""
    try:
        await log_bot.send_message(admin_id, ai.fit(redact(text)))
    except AiogramError as error:
        logging.warning("Лог-бот не смог написать (%s) — напиши ему /start", error)


async def try_send(bot, chat_id, text, **kwargs):
    """Сообщение пользователю: флуд-лимит — пауза и один повтор. -> (Message, None) или (None, ошибка)."""
    try:
        try:
            return await bot.send_message(chat_id, text, **kwargs), None
        except TelegramRetryAfter as error:
            await asyncio.sleep(error.retry_after)
            return await bot.send_message(chat_id, text, **kwargs), None
    except AiogramError as error:
        logging.warning("Не смог отправить сообщение в чат %s: %s", chat_id, error)
        return None, error


async def send(bot, chat_id, text, **kwargs):
    """Как try_send, но только Message или None — когда подробности ошибки не нужны."""
    return (await try_send(bot, chat_id, text, **kwargs))[0]


async def serve(poll, log_bot, admin_id, models, marker, retry_delay=10):
    """Запускает poll() и сообщает о запуске, штатной остановке и падении.

    marker — файл-метка «бот работает». Остался с прошлого запуска — значит, тот оборвался так,
    что сообщить было некому (процесс убили, сервер упал): говорим об этом при запуске."""
    text = f"Бот запущен. Модели: {', '.join(models)}"
    if os.path.exists(marker):
        text += "\n" + CRASH_NOTE
    await tell(log_bot, admin_id, text)
    with open(marker, "w"):
        pass
    # через logging, а не print: print в консоль не в UTF-8 падает на кириллице и уронил бы бота
    logging.info("\n%s", banner(models, datetime.now(timezone.utc)))
    delay = retry_delay
    try:
        while True:
            try:
                await poll()
                break
            except TelegramNetworkError as error:
                # нет сети при старте (getMe) — ждём и пробуем снова, пауза растёт до 5 минут
                logging.warning("Нет связи с Telegram (%s), повтор через %s с", error, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 300)
    except Exception as error:
        await tell(log_bot, admin_id, f"Бот упал: {type(error).__name__}: {error}")
        raise
    else:
        # Stop/Restart на хостинге присылает сигнал — aiogram штатно завершает polling
        await tell(log_bot, admin_id, STOPPED_TEXT)
    finally:
        with contextlib.suppress(OSError):
            os.remove(marker)


async def delete(bot, chat_id, ids):
    """Удаляет сообщения пачками; пачка не прошла — по одному, ошибки глушим (старше 48 ч Telegram не удаляет)."""
    for i in range(0, len(ids), DELETE_BATCH):
        batch = ids[i:i + DELETE_BATCH]
        try:
            await bot.delete_messages(chat_id, batch)
        except AiogramError:
            for message_id in batch:
                with contextlib.suppress(AiogramError):
                    await bot.delete_message(chat_id, message_id)


def build_dispatcher(allowed, admin_id, log_bot, ask, creator_name, creator_url):
    """ask — async (history, question) -> (ответ, модель). История — в памяти, пропадает при перезапуске."""
    allowed = set(allowed) | {admin_id}
    # упрощение: всё в памяти — после перезапуска старые сообщения не удалятся и история пропадёт; нужно — хранить в файле
    histories = {}  # id пользователя -> deque пар (вопрос, ответ)
    locks = {}  # вопросы одного пользователя — строго по очереди, иначе история перепутается
    # меню и очистка одного чата — по очереди. Порядок захвата всегда locks -> ui_locks, иначе взаимная блокировка
    ui_locks = {}
    tracked = {}  # id чата -> пары (id сообщения, время now()) входящих и наших, которые удалим при очистке
    menus = {}  # id чата -> id текущего сообщения меню
    status = {}  # id чата -> id «Сессия очищена.» / «Сессия уже пуста.»: следующее «Очистить сессию» его удалит
    # упрощение: словарь растёт с каждым новым чужим id; если чужих станет очень много — чистить старые записи
    denied_at = {}  # id чужого -> когда ему последний раз ответили «Нет доступа»
    dp = Dispatcher()
    dp.message.filter(F.chat.type == "private")  # в группах молчим
    dp.callback_query.filter(F.message.chat.type == "private")

    def track(message):
        """Запоминает сообщение для очистки; None (не отправилось) пропускает."""
        if message is not None:
            tracked.setdefault(message.chat.id, []).append((message.message_id, now()))
        return message

    async def drop(bot, chat_id, message_id):
        """Удаляет одно сообщение и забывает его; ошибка удаления не страшна."""
        with contextlib.suppress(AiogramError):
            await bot.delete_message(chat_id, message_id)
        if chat_id in tracked:
            tracked[chat_id] = [pair for pair in tracked[chat_id] if pair[0] != message_id]

    async def show_menu(bot, chat_id):
        """Меню всегда одно: старое удаляем, новое шлём вниз. Вызывать под ui_locks[chat_id]."""
        if chat_id in menus:
            await drop(bot, chat_id, menus.pop(chat_id))
        sent = track(await send(bot, chat_id, menu_text(creator_name, creator_url),
                                parse_mode="HTML", reply_markup=menu_kb(creator_url)))
        if sent is not None:
            menus[chat_id] = sent.message_id

    async def log(text):
        try:
            await log_bot.send_message(admin_id, ai.fit(redact(text)))
        except AiogramError as error:
            # сломанный лог не должен мешать ответу пользователю
            logging.warning("Лог-бот не смог отправить сообщение: %s", error)

    @dp.message(lambda message: message.from_user.id not in allowed)
    async def denied(message, bot):
        user_id = message.from_user.id
        if user_id in denied_at and now() - denied_at[user_id] < DENIED_EVERY:
            return  # уже отвечали недавно — молчим, чтобы не спамить ни его, ни владельца
        denied_at[user_id] = now()
        await send(bot, message.chat.id, DENIED_TEXT)
        await log(f"Нет доступа: {who(message.from_user)}\n\n{message.text or '(не текст)'}")

    @dp.message(CommandStart())
    async def start(message, bot):
        track(message)
        async with ui_locks.setdefault(message.chat.id, asyncio.Lock()):
            await show_menu(bot, message.chat.id)

    @dp.message(F.text == MENU_BUTTON)
    async def menu(message, bot):
        async with ui_locks.setdefault(message.chat.id, asyncio.Lock()):
            await drop(bot, message.chat.id, message.message_id)
            await show_menu(bot, message.chat.id)

    @dp.message(F.text == CLEAR_BUTTON)
    async def clear(message, bot):
        chat_id = message.chat.id
        async with ui_locks.setdefault(chat_id, asyncio.Lock()):
            await drop(bot, chat_id, message.message_id)
            # «пусто» — по истории: «Сессия очищена.» и меню отслеживаются всегда, по ним судить нельзя
            if not histories.get(message.from_user.id):
                if chat_id in status:
                    await drop(bot, chat_id, status.pop(chat_id))
                sent = track(await send(bot, chat_id, EMPTY_TEXT))
                if sent is not None:
                    status[chat_id] = sent.message_id
            else:
                track(await send(bot, chat_id, CONFIRM_TEXT, reply_markup=CONFIRM_KB))

    @dp.message(F.text)
    async def question(message, bot):
        user, text, chat_id = message.from_user, message.text, message.chat.id
        track(message)
        # ui_lock — только на удаление меню и отпускаем до locks: ждать locks, держа ui_lock, нельзя
        async with ui_locks.setdefault(chat_id, asyncio.Lock()):
            if chat_id in menus:  # начался разговор — меню больше не нужно
                await drop(bot, chat_id, menus.pop(chat_id))
        lock = locks.setdefault(user.id, asyncio.Lock())
        if lock.locked():  # без очереди: ещё думаем над прошлым вопросом
            track(await send(bot, chat_id, BUSY_TEXT))
            return
        async with lock:
            history = histories.setdefault(user.id, deque(maxlen=HISTORY_SIZE))
            started = time.monotonic()
            try:
                async with typing(bot, chat_id):
                    answer, model = await ask(list(history), text)
            except Exception as error:
                logging.exception("Gemini не ответил")
                reply = BLOCKED_TEXT if isinstance(error, ai.BlockedError) else ERROR_TEXT
                track(await send(bot, chat_id, reply, reply_markup=MAIN_KB))
                await log(f"Ошибка: {who(user)}\n\n{text}\n\n{type(error).__name__}: {error}")
                return
            sent, error = await try_send(bot, chat_id, answer, reply_markup=MAIN_KB)
            if sent is None:  # пользователь ответа не увидел — в историю не пишем
                await log(f"Не смог отправить ответ: {who(user)}\n\n{text}\n\n{type(error).__name__}: {error}")
                return
            track(sent)
            history.append((text, answer))
        await log(f"Вопрос: {who(user)}\n\n{text}\n\n"
                  f"Ответ ({model}, {time.monotonic() - started:.1f} с):\n{short(answer)}")

    @dp.message()
    async def other(message, bot):
        track(message)
        track(await send(bot, message.chat.id, ONLY_TEXT))

    @dp.callback_query(lambda callback: callback.from_user.id not in allowed)
    async def denied_button(callback):
        await callback.answer(DENIED_TEXT)

    @dp.callback_query(F.data == "clear:no")
    async def clear_no(callback, bot):
        with contextlib.suppress(AiogramError):
            await callback.answer()
        chat_id = callback.message.chat.id
        async with ui_locks.setdefault(chat_id, asyncio.Lock()):
            if callback.message.message_id != status.get(chat_id):  # иначе оно уже стало «Сессия очищена.»
                await drop(bot, chat_id, callback.message.message_id)

    @dp.callback_query(F.data == "clear:yes")
    async def clear_yes(callback, bot):
        with contextlib.suppress(AiogramError):
            await callback.answer()
        user, chat_id, warning = callback.from_user, callback.message.chat.id, callback.message.message_id
        # сначала дождаться текущего ответа Gemini, иначе он придёт уже в очищенный чат
        async with locks.setdefault(user.id, asyncio.Lock()):
            async with ui_locks.setdefault(chat_id, asyncio.Lock()):
                if warning == status.get(chat_id):
                    return  # повторное нажатие: предупреждение уже стало «Сессия очищена.»
                if warning not in [message_id for message_id, _ in tracked.get(chat_id, [])]:
                    # устаревшее предупреждение: кнопка из прошлого запуска
                    await drop(bot, chat_id, warning)
                    return
                histories.pop(user.id, None)
                # удаляем всё, кроме предупреждения; старше DELETE_AGE Telegram всё равно не удалит — не пробуем
                pairs = tracked.pop(chat_id, [])
                tracked[chat_id] = [pair for pair in pairs if pair[0] == warning]
                await delete(bot, chat_id, [message_id for message_id, at in pairs
                                            if message_id != warning and now() - at < DELETE_AGE])
                menus.pop(chat_id, None)
                # упрощение: нижние кнопки держало удалённое сообщение — на ПК они пропадут до следующего ответа
                try:
                    # без reply_markup Telegram убирает кнопки Принять/Отменить
                    await bot.edit_message_text(CLEARED_TEXT, chat_id=chat_id, message_id=warning)
                    status[chat_id] = warning
                except AiogramError:
                    await drop(bot, chat_id, warning)
                    sent = track(await send(bot, chat_id, CLEARED_TEXT, reply_markup=MAIN_KB))
                    if sent is not None:
                        status[chat_id] = sent.message_id
                await show_menu(bot, chat_id)  # меню — последним
        await log(f"Очистил сессию: {who(user)}")

    return dp


async def main():
    log_format = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    for handler in logging.getLogger().handlers:  # токены ботов не должны попасть в консоль хостинга
        handler.setFormatter(RedactingFormatter(log_format))
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)  # без строки на каждый апдейт
    here = os.path.dirname(os.path.abspath(__file__))
    load_env(os.path.join(here, ".env"))
    settings = read_settings(os.environ)
    admin_id, models = settings["admin_id"], settings["models"]

    async with aiohttp.ClientSession() as session, \
            Bot(settings["token"]) as tg, Bot(settings["log_token"]) as log_bot:
        weekly = asyncio.create_task(weekly_loop(log_bot, admin_id))  # ссылку держим, иначе задачу соберёт GC
        ask = functools.partial(ai.ask, session, settings["api_key"], models)
        dp = build_dispatcher(settings["allowed"], admin_id, log_bot, ask,
                              settings["creator_name"], settings["creator_url"])
        await serve(lambda: dp.start_polling(tg), log_bot, admin_id, models, os.path.join(here, ".running"))


if __name__ == "__main__":
    asyncio.run(main())
