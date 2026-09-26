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
from aiogram.exceptions import TelegramAPIError
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
LOG_ANSWER = 300  # сколько символов ответа показывать в логе
DELETE_BATCH = 100  # больше id за раз delete_messages не принимает
MSK = timezone(timedelta(hours=3))  # в Москве нет перехода на летнее время


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
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
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
            with contextlib.suppress(TelegramAPIError):
                await bot.send_chat_action(chat_id, "typing")

    with contextlib.suppress(TelegramAPIError):
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
        now = datetime.now(timezone.utc)
        reset = next_weekly_reset(now)
        # +5 с — проснуться точно после сброса, иначе при ранней побудке пришло бы дважды
        await asyncio.sleep((reset - now).total_seconds() + 5)
        try:
            await log_bot.send_message(admin_id, weekly_text(reset))
        except TelegramAPIError as error:
            logging.warning("Лог-бот не смог сообщить о сбросе лимита: %s", error)


async def delete(bot, chat_id, ids):
    """Удаляет сообщения пачками; пачка не прошла — по одному, ошибки глушим (старше 48 ч Telegram не удаляет)."""
    for i in range(0, len(ids), DELETE_BATCH):
        batch = ids[i:i + DELETE_BATCH]
        try:
            await bot.delete_messages(chat_id, batch)
        except TelegramAPIError:
            for message_id in batch:
                with contextlib.suppress(TelegramAPIError):
                    await bot.delete_message(chat_id, message_id)


def build_dispatcher(allowed, admin_id, log_bot, ask, creator_name, creator_url):
    """ask — async (history, question) -> (ответ, модель). История — в памяти, пропадает при перезапуске."""
    allowed = set(allowed) | {admin_id}
    # упрощение: всё в памяти — после перезапуска старые сообщения не удалятся и история пропадёт; нужно — хранить в файле
    histories = {}  # id пользователя -> deque пар (вопрос, ответ)
    locks = {}  # вопросы одного пользователя — строго по очереди, иначе история перепутается
    tracked = {}  # id чата -> id сообщений (входящих и наших), которые удалим при очистке
    menus = {}  # id чата -> id текущего сообщения меню
    dp = Dispatcher()
    dp.message.filter(F.chat.type == "private")  # в группах молчим
    dp.callback_query.filter(F.message.chat.type == "private")

    def track(message):
        tracked.setdefault(message.chat.id, []).append(message.message_id)
        return message

    async def drop(bot, chat_id, message_id):
        """Удаляет одно сообщение и забывает его; ошибка удаления не страшна."""
        with contextlib.suppress(TelegramAPIError):
            await bot.delete_message(chat_id, message_id)
        with contextlib.suppress(ValueError):
            tracked.get(chat_id, []).remove(message_id)

    async def show_menu(bot, chat_id):
        """Меню всегда одно: старое удаляем, новое шлём вниз."""
        if chat_id in menus:
            await drop(bot, chat_id, menus.pop(chat_id))
        sent = await bot.send_message(chat_id, menu_text(creator_name, creator_url),
                                      parse_mode="HTML", reply_markup=menu_kb(creator_url))
        menus[chat_id] = track(sent).message_id

    async def log(text):
        try:
            await log_bot.send_message(admin_id, ai.fit(text))
        except TelegramAPIError as error:
            # сломанный лог не должен мешать ответу пользователю
            logging.warning("Лог-бот не смог отправить сообщение: %s", error)

    @dp.message(lambda message: message.from_user.id not in allowed)
    async def denied(message):
        await message.answer(DENIED_TEXT)
        await log(f"Нет доступа: {who(message.from_user)}\n\n{message.text or '(не текст)'}")

    @dp.message(CommandStart())
    async def start(message, bot):
        track(message)
        await show_menu(bot, message.chat.id)

    @dp.message(F.text == MENU_BUTTON)
    async def menu(message, bot):
        await drop(bot, message.chat.id, message.message_id)
        await show_menu(bot, message.chat.id)

    @dp.message(F.text == CLEAR_BUTTON)
    async def clear(message, bot):
        await drop(bot, message.chat.id, message.message_id)
        if not histories.get(message.from_user.id):
            track(await message.answer(EMPTY_TEXT))
        else:
            track(await message.answer(CONFIRM_TEXT, reply_markup=CONFIRM_KB))

    @dp.message(F.text)
    async def question(message, bot):
        user, text, chat_id = message.from_user, message.text, message.chat.id
        track(message)
        if chat_id in menus:  # начался разговор — меню больше не нужно
            await drop(bot, chat_id, menus.pop(chat_id))
        async with locks.setdefault(user.id, asyncio.Lock()):
            history = histories.setdefault(user.id, deque(maxlen=HISTORY_SIZE))
            started = time.monotonic()
            try:
                async with typing(bot, chat_id):
                    answer, model = await ask(list(history), text)
            except Exception as error:
                logging.exception("Gemini не ответил")
                track(await message.answer(ERROR_TEXT, reply_markup=MAIN_KB))
                await log(f"Ошибка: {who(user)}\n\n{text}\n\n{type(error).__name__}: {error}")
                return
            track(await message.answer(answer, reply_markup=MAIN_KB))
            history.append((text, answer))
        await log(f"Вопрос: {who(user)}\n\n{text}\n\n"
                  f"Ответ ({model}, {time.monotonic() - started:.1f} с):\n{short(answer)}")

    @dp.message()
    async def other(message):
        track(message)
        track(await message.answer(ONLY_TEXT))

    @dp.callback_query(lambda callback: callback.from_user.id not in allowed)
    async def denied_button(callback):
        await callback.answer(DENIED_TEXT)

    @dp.callback_query(F.data == "clear:no")
    async def clear_no(callback, bot):
        await drop(bot, callback.message.chat.id, callback.message.message_id)
        await callback.answer()

    @dp.callback_query(F.data == "clear:yes")
    async def clear_yes(callback, bot):
        user, chat_id = callback.from_user, callback.message.chat.id
        histories.pop(user.id, None)
        await delete(bot, chat_id, tracked.pop(chat_id, []))  # предупреждение тоже там
        menus.pop(chat_id, None)
        # это сообщение держит нижние кнопки: без него на ПК они пропадут вместе со старыми сообщениями
        track(await bot.send_message(chat_id, CLEARED_TEXT, reply_markup=MAIN_KB))
        await show_menu(bot, chat_id)  # меню — последним
        await callback.answer()
        await log(f"Очистил сессию: {who(user)}")

    return dp


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    names = ("BOT_TOKEN", "LOG_BOT_TOKEN", "ADMIN_ID", "GEMINI_API_KEY", "CREATOR_USERNAME")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise SystemExit("Не заданы в .env: " + ", ".join(missing))
    admin = parse_ids(os.environ["ADMIN_ID"])
    if len(admin) != 1:
        raise SystemExit("ADMIN_ID — ровно один Telegram id")
    admin_id = admin.pop()
    allowed = parse_ids(os.environ.get("ALLOWED_IDS"))
    models = [m.strip() for m in os.environ.get("GEMINI_MODELS", "").split(",") if m.strip()] or ai.DEFAULT_MODELS
    username = os.environ["CREATOR_USERNAME"].lstrip("@")
    creator_name = os.environ.get("CREATOR_NAME") or "@" + username
    creator_url = f"https://t.me/{username}"

    async with aiohttp.ClientSession() as session, \
            Bot(os.environ["BOT_TOKEN"]) as tg, Bot(os.environ["LOG_BOT_TOKEN"]) as log_bot:
        try:
            await log_bot.send_message(admin_id, f"Бот запущен. Модели: {', '.join(models)}")
        except TelegramAPIError as error:
            logging.warning("Лог-бот не может написать тебе (%s) — напиши ему /start", error)
        weekly = asyncio.create_task(weekly_loop(log_bot, admin_id))  # ссылку держим, иначе задачу соберёт GC
        ask = functools.partial(ai.ask, session, os.environ["GEMINI_API_KEY"], models)
        await build_dispatcher(allowed, admin_id, log_bot, ask, creator_name, creator_url).start_polling(tg)


if __name__ == "__main__":
    asyncio.run(main())
