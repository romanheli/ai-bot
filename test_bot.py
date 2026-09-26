import asyncio
import logging
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from aiogram import Bot
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import (AnswerCallbackQuery, DeleteMessage, DeleteMessages, GetMe, SendChatAction,
                             SendMessage)
from aiogram.types import InlineKeyboardMarkup, ReplyKeyboardMarkup, Update

import ai
import bot
from testutil import (ADMIN, NAME, OTHER, OTHER2, PHOTO, STICKER, URL, USER, USER2, FakeAsk, MockedSession,
                      callback, reset_ids, rows, start, update)

GROUP = {"id": -100123, "type": "group", "title": "Группа"}


def inline(markup):
    """Инлайн-кнопки по рядам: (надпись, callback_data или url)."""
    return [[(b.text, b.callback_data or b.url) for b in row] for row in markup.inline_keyboard]


class MenuTextTest(unittest.TestCase):
    def test_text(self):
        self.assertEqual(bot.menu_text(NAME, URL),
                         "Привет, я Epsilon, ИИ бот помощник\nТы можешь спросить меня о чем угодно\n\n"
                         '<i>создатель - <a href="https://t.me/author">Автор</a></i>')

    def test_escape(self):
        self.assertIn("&lt;A&amp;B&gt;", bot.menu_text("<A&B>", URL))


class WeeklyResetTest(unittest.TestCase):
    def check(self, now, expected):
        self.assertEqual(bot.next_weekly_reset(now), expected)

    def test_midweek(self):
        # среда -> ближайшее воскресенье 00:00 UTC
        self.check(datetime(2026, 9, 23, 15, 30, tzinfo=timezone.utc), datetime(2026, 9, 27, tzinfo=timezone.utc))

    def test_saturday_night(self):
        self.check(datetime(2026, 9, 26, 23, 59, tzinfo=timezone.utc), datetime(2026, 9, 27, tzinfo=timezone.utc))

    def test_sunday_after_reset(self):
        # сброс уже был сегодня — следующий через неделю
        self.check(datetime(2026, 9, 27, 0, 0, tzinfo=timezone.utc), datetime(2026, 10, 4, tzinfo=timezone.utc))
        self.check(datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc), datetime(2026, 10, 4, tzinfo=timezone.utc))

    def test_other_timezone(self):
        # вс 02:00 по Москве = сб 23:00 UTC — сброс ещё впереди
        msk = timezone(timedelta(hours=3))
        self.check(datetime(2026, 9, 27, 2, 0, tzinfo=msk), datetime(2026, 9, 27, tzinfo=timezone.utc))


class WeeklyTextTest(unittest.TestCase):
    def test_moscow_time(self):
        # вс 00:00 UTC = вс 03:00 МСК
        self.assertEqual(bot.weekly_text(datetime(2026, 9, 27, tzinfo=timezone.utc)),
                         "Недельный лимит Claude сбросился: 27.09.2026 03:00 МСК")

    def test_padding(self):
        self.assertEqual(bot.fmt_msk(datetime(2026, 1, 4, 21, 5, tzinfo=timezone.utc)), "05.01.2026 00:05")


class WeeklyLoopTest(unittest.IsolatedAsyncioTestCase):
    """weekly_loop на подменных часах: сон не ждёт, а только двигает время."""

    async def asyncSetUp(self):
        self.log_session = MockedSession()
        self.log_bot = Bot("43:TEST", session=self.log_session)
        self.clock = datetime(2026, 9, 26, 23, 0, tzinfo=timezone.utc)  # сб 23:00 UTC, сброс через час
        self.sleeps = []

    def logs(self):
        return [m.text for m in self.log_session.requests if isinstance(m, SendMessage) and m.chat_id == ADMIN]

    async def run_loop(self, early=0):
        """Крутит weekly_loop до первого сообщения лог-бота (или 10 снов) и останавливает его.
        early — на столько секунд раньше срока просыпается первый сон."""
        test = self

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return test.clock

        async def fake_sleep(delay, *args, **kwargs):
            if not delay:  # asyncio.sleep(0) из MockedSession
                return
            if self.logs() or len(self.sleeps) >= 10:
                raise asyncio.CancelledError
            self.sleeps.append(delay)
            self.clock += timedelta(seconds=delay - (early if len(self.sleeps) == 1 else 0))

        with patch("bot.datetime", Clock), patch("asyncio.sleep", fake_sleep):
            with self.assertRaises(asyncio.CancelledError):
                await bot.weekly_loop(self.log_bot, ADMIN)

    async def test_early_wakeup(self):
        await self.run_loop(early=60)
        reset = datetime(2026, 9, 27, tzinfo=timezone.utc)
        self.assertEqual(self.logs(), [bot.weekly_text(reset)])
        # проснулись на минуту раньше — доспали и написали уже после сброса
        self.assertEqual(len(self.sleeps), 2)
        self.assertGreaterEqual(self.clock, reset)

    async def test_error_does_not_kill_loop(self):
        real = bot.next_weekly_reset
        calls = []

        def flaky(now):
            calls.append(now)
            if len(calls) == 1:
                raise RuntimeError("сбой")
            return real(now)

        with patch("bot.next_weekly_reset", side_effect=flaky), self.assertLogs(level="ERROR"):
            await self.run_loop()
        self.assertEqual(self.logs(), [bot.weekly_text(datetime(2026, 9, 27, tzinfo=timezone.utc))])


class BannerTest(unittest.TestCase):
    def test_banner(self):
        text = bot.banner(["m1", "m2"], datetime(2026, 9, 27, 11, 5, tzinfo=timezone.utc))
        self.assertIn("EPSILON ЗАПУЩЕН", text)
        self.assertIn("27.09.2026 14:05 МСК", text)
        self.assertIn("m1, m2", text)
        # цветной: зелёный в начале, сброс цвета в конце
        self.assertTrue(text.startswith("\033[1;92m"))
        self.assertTrue(text.endswith("\033[0m"))


class ParseIdsTest(unittest.TestCase):
    def test_mixed_separators(self):
        self.assertEqual(bot.parse_ids("123, 456 789"), {123, 456, 789})

    def test_commas_only(self):
        self.assertEqual(bot.parse_ids("1,2"), {1, 2})

    def test_empty(self):
        self.assertEqual(bot.parse_ids(""), set())
        self.assertEqual(bot.parse_ids(None), set())

    def test_bad(self):
        with self.assertRaises(SystemExit):
            bot.parse_ids("12a")


class LoadEnvTest(unittest.TestCase):
    """load_env пишет в os.environ — после каждого теста окружение возвращается как было."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, ".env")
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for key in [k for k in os.environ if k.startswith("EPS_")]:
            del os.environ[key]

    def load(self, text, encoding="utf-8"):
        with open(self.path, "w", encoding=encoding) as f:
            f.write(text)
        bot.load_env(self.path)

    def test_bom(self):
        # Блокнот пишет BOM в начало файла
        self.load("EPS_A=1\n", encoding="utf-8-sig")
        self.assertEqual(os.environ.get("EPS_A"), "1")

    def test_comments(self):
        self.load("# EPS_B=1\n\n   # EPS_C=2\nEPS_D=3\n")
        self.assertNotIn("EPS_B", os.environ)
        self.assertNotIn("EPS_C", os.environ)
        self.assertEqual(os.environ.get("EPS_D"), "3")

    def test_quotes(self):
        self.load("EPS_E=\"a # b\"\nEPS_F='c'\n")
        self.assertEqual(os.environ.get("EPS_E"), "a # b")
        self.assertEqual(os.environ.get("EPS_F"), "c")

    def test_inline_comment(self):
        self.load("EPS_G=abc #комментарий\nEPS_H=a#b\nEPS_I=  x y  \n")
        self.assertEqual(os.environ.get("EPS_G"), "abc")
        self.assertEqual(os.environ.get("EPS_H"), "a#b")  # без пробела перед # — не комментарий
        self.assertEqual(os.environ.get("EPS_I"), "x y")

    def test_export(self):
        self.load("export EPS_J=1\nexport EPS_K=\"q\"\n")
        self.assertEqual(os.environ.get("EPS_J"), "1")
        self.assertEqual(os.environ.get("EPS_K"), "q")
        self.assertNotIn("export EPS_J", os.environ)

    def test_empty_key(self):
        self.load("=value\n  = x\nEPS_L=1\n")  # не должно упасть
        self.assertNotIn("", os.environ)
        self.assertEqual(os.environ.get("EPS_L"), "1")

    def test_keeps_existing(self):
        os.environ["EPS_M"] = "old"
        self.load("EPS_M=new\n")
        self.assertEqual(os.environ["EPS_M"], "old")

    def test_no_file(self):
        bot.load_env(os.path.join(self.dir.name, "нет.env"))  # не должно упасть


class ReadSettingsTest(unittest.TestCase):
    ENV = {"BOT_TOKEN": "1:a", "LOG_BOT_TOKEN": "2:b", "ADMIN_ID": "999999", "GEMINI_API_KEY": "key",
           "CREATOR_USERNAME": "@author"}

    def read(self, **changes):
        """read_settings от ENV с изменениями; значение None — убрать переменную."""
        env = {k: v for k, v in dict(self.ENV, **changes).items() if v is not None}
        return bot.read_settings(env)

    def error(self, **changes):
        with self.assertRaises(SystemExit) as cm:
            self.read(**changes)
        return str(cm.exception)

    def test_full(self):
        self.assertEqual(self.read(ALLOWED_IDS="1, 2 3"), {
            "token": "1:a", "log_token": "2:b", "admin_id": 999999, "allowed": {1, 2, 3}, "api_key": "key",
            "models": ai.DEFAULT_MODELS, "creator_name": "@author", "creator_url": "https://t.me/author"})

    def test_no_allowed(self):
        self.assertEqual(self.read()["allowed"], set())

    def test_missing(self):
        text = self.error(BOT_TOKEN=None, GEMINI_API_KEY="")
        self.assertIn("BOT_TOKEN", text)
        self.assertIn("GEMINI_API_KEY", text)
        self.assertNotIn("LOG_BOT_TOKEN", text)
        self.assertNotIn("CREATOR_USERNAME", text)

    def test_admin_not_one(self):
        self.error(ADMIN_ID="1, 2")

    def test_bad_ids(self):
        self.error(ADMIN_ID="abc")
        self.error(ALLOWED_IDS="1, x")

    def test_models(self):
        self.assertEqual(self.read(GEMINI_MODELS="a, ,b")["models"], ["a", "b"])
        self.assertEqual(self.read(GEMINI_MODELS="")["models"], ai.DEFAULT_MODELS)
        self.assertEqual(self.read(GEMINI_MODELS=" , ")["models"], ai.DEFAULT_MODELS)

    def test_creator(self):
        settings = self.read(CREATOR_USERNAME="x")
        self.assertEqual(settings["creator_url"], "https://t.me/x")
        self.assertEqual(settings["creator_name"], "@x")
        self.assertEqual(self.read(CREATOR_NAME="Автор")["creator_name"], "Автор")


class RedactTest(unittest.TestCase):
    def test_redact(self):
        self.assertEqual(bot.redact("https://api.telegram.org/bot123456:AAE_x-9z/getMe и bot7:q"),
                         "https://api.telegram.org/bot<TOKEN>/getMe и bot<TOKEN>")

    def test_no_token(self):
        text = "@botfather, bot: 12"
        self.assertEqual(bot.redact(text), text)

    def test_formatter_traceback(self):
        try:
            raise RuntimeError("POST https://api.telegram.org/bot123:SeCrEt_1/sendMessage")
        except RuntimeError:
            record = logging.LogRecord("t", logging.ERROR, __file__, 1, "сбой bot42:SeCrEt_2", None, sys.exc_info())
        text = bot.RedactingFormatter("%(levelname)s %(message)s").format(record)
        self.assertNotIn("SeCrEt", text)
        self.assertIn("ERROR сбой bot<TOKEN>", text)
        self.assertIn("Traceback", text)
        self.assertIn("RuntimeError: POST https://api.telegram.org/bot<TOKEN>/sendMessage", text)


class ServeTest(unittest.IsolatedAsyncioTestCase):
    """serve: сообщения лог-бота о запуске, остановке и падении + файл-метка «бот работает»."""

    async def asyncSetUp(self):
        self.log_session = MockedSession()
        self.log_bot = Bot("43:TEST", session=self.log_session)
        self.dir = tempfile.TemporaryDirectory()
        self.marker = os.path.join(self.dir.name, ".running")
        self.marker_during_poll = None

    async def asyncTearDown(self):
        self.dir.cleanup()

    def logs(self):
        return [m.text for m in self.log_session.requests if isinstance(m, SendMessage) and m.chat_id == ADMIN]

    async def poll(self):
        self.marker_during_poll = os.path.exists(self.marker)

    async def serve(self, poll=None, **kwargs):
        await bot.serve(poll or self.poll, self.log_bot, ADMIN, ["m1", "m2"], self.marker, **kwargs)

    def flaky_poll(self, failures):
        """poll, который первые failures раз падает без сети, потом штатно завершается; вызовы — в self.polls."""
        self.polls = 0

        async def poll():
            self.polls += 1
            if self.polls <= failures:
                raise TelegramNetworkError(GetMe(), "нет сети")

        return poll

    async def test_clean_stop(self):
        await self.serve()
        logs = self.logs()
        self.assertEqual(len(logs), 2)
        self.assertIn("Бот запущен", logs[0])
        self.assertIn("m1, m2", logs[0])
        self.assertNotIn(bot.CRASH_NOTE, logs[0])
        self.assertEqual(logs[1], bot.STOPPED_TEXT)
        # пока работает — метка есть, после штатной остановки — нет
        self.assertTrue(self.marker_during_poll)
        self.assertFalse(os.path.exists(self.marker))

    async def test_crash(self):
        async def boom():
            raise RuntimeError("всё сломалось")

        with self.assertRaises(RuntimeError):
            await self.serve(boom)
        self.assertEqual(self.logs()[-1], "Бот упал: RuntimeError: всё сломалось")
        self.assertNotIn(bot.STOPPED_TEXT, self.logs())
        # о падении уже сообщили — следующий запуск не должен сообщать ещё раз
        self.assertFalse(os.path.exists(self.marker))

    async def test_killed_last_time(self):
        open(self.marker, "w").close()  # прошлый запуск не убрал метку — его убили
        await self.serve()
        self.assertIn(bot.CRASH_NOTE, self.logs()[0])

    async def test_log_bot_broken(self):
        self.log_session.fail = True
        await self.serve()  # исключение не должно вылететь
        self.assertTrue(self.marker_during_poll)

    async def test_log_bot_decode_error(self):
        # ClientDecodeError — не TelegramAPIError, но тоже не должен ронять бота
        self.log_session.decode_error = True
        await self.serve()
        self.assertTrue(self.marker_during_poll)

    async def test_crash_redacted(self):
        async def boom():
            raise RuntimeError("https://api.telegram.org/bot123:SeCrEt/getMe")

        with self.assertRaises(RuntimeError):
            await self.serve(boom)
        self.assertEqual(self.logs()[-1], "Бот упал: RuntimeError: https://api.telegram.org/bot<TOKEN>/getMe")

    async def test_tell_redact_and_fit(self):
        await bot.tell(self.log_bot, ADMIN, "bot1:SeCrEt " + "я" * 5000)
        [text] = self.logs()
        self.assertTrue(text.startswith("bot<TOKEN> я"))
        self.assertNotIn("SeCrEt", text)
        self.assertLessEqual(ai.tg_len(text), 4096)

    async def test_network_retry(self):
        # нет сети при старте (getMe) — ждём и пробуем снова, «Бот упал» не шлём
        poll = self.flaky_poll(2)
        with self.assertLogs(level="INFO") as cm:
            await self.serve(poll, retry_delay=0)
        self.assertEqual(self.polls, 3)
        logs = self.logs()
        self.assertEqual(len(logs), 2)
        self.assertIn("Бот запущен", logs[0])
        self.assertEqual(logs[1], bot.STOPPED_TEXT)
        self.assertFalse(any("Бот упал" in text for text in logs))
        # баннер — один раз, до повторов; каждая неудача — предупреждение в консоль
        self.assertEqual(sum("EPSILON ЗАПУЩЕН" in r.getMessage() for r in cm.records), 1)
        self.assertEqual(sum(r.levelno == logging.WARNING for r in cm.records), 2)
        self.assertFalse(os.path.exists(self.marker))

    async def test_network_retry_backoff(self):
        delays = []

        async def fake_sleep(delay, *args, **kwargs):
            if delay:  # asyncio.sleep(0) из MockedSession не считаем
                delays.append(delay)

        poll = self.flaky_poll(7)
        with patch("asyncio.sleep", fake_sleep), self.assertLogs(level="WARNING"):
            await self.serve(poll, retry_delay=10)
        self.assertEqual(self.polls, 8)
        self.assertEqual(delays, [10, 20, 40, 80, 160, 300, 300])
        self.assertEqual(self.logs()[-1], bot.STOPPED_TEXT)


class BotCase(unittest.IsolatedAsyncioTestCase):
    """Основа тестов через build_dispatcher: основной бот и лог-бот — разные Bot с разными MockedSession."""

    async def asyncSetUp(self):
        # фейковые токены правильного формата — в сеть не ходим
        reset_ids()
        self.session = MockedSession()
        self.tg = Bot("42:TEST", session=self.session)
        self.log_session = MockedSession()
        self.log_bot = Bot("43:TEST", session=self.log_session)
        self.ask = FakeAsk()
        self.dp = self.make()

    async def asyncTearDown(self):
        await self.tg.session.close()
        await self.log_bot.session.close()

    def make(self, allowed=frozenset({USER, USER2})):
        return bot.build_dispatcher(set(allowed), ADMIN, self.log_bot, self.ask, NAME, URL)

    async def push(self, data, dp=None):
        # тайм-аут: сломанная блокировка должна ронять тест, а не вешать весь прогон
        await asyncio.wait_for((dp or self.dp).feed_update(
            self.tg, Update.model_validate(data, context={"bot": self.tg})), timeout=5)
        # если лог шлётся фоновой задачей — даём ей доработать
        for _ in range(10):
            await asyncio.sleep(0)

    def sent(self):
        """Сообщения основного бота (SendMessage) по порядку."""
        return [m for m in self.session.requests if isinstance(m, SendMessage)]

    def sent_ids(self, text):
        """Id, выданные отправленным сообщениям с этим текстом, по порядку."""
        return [i for m, i in self.session.ids if m.text == text]

    def deletes(self):
        """Запросы удаления основного бота (DeleteMessage и DeleteMessages)."""
        return [m for m in self.session.requests if isinstance(m, (DeleteMessage, DeleteMessages))]

    def deleted(self):
        """Множество id, которые бот пытался удалить."""
        ids = set()
        for m in self.deletes():
            ids.update(m.message_ids if isinstance(m, DeleteMessages) else [m.message_id])
        return ids

    def answered(self):
        """Бот ответил на нажатие инлайн-кнопки (AnswerCallbackQuery)."""
        return any(isinstance(m, AnswerCallbackQuery) for m in self.session.requests)

    def logs(self):
        """Тексты сообщений лог-бота в чат ADMIN."""
        return [m.text for m in self.log_session.requests if isinstance(m, SendMessage) and m.chat_id == ADMIN]

    def assertLogged(self, *parts):
        """Хотя бы одно сообщение лог-бота содержит все parts."""
        self.assertTrue(any(all(p in text for p in parts) for text in self.logs()),
                        f"нет лога с {parts}: {self.logs()}")

    async def until(self, condition):
        """Отдаёт управление другим задачам, пока condition() не станет истинным."""
        for _ in range(200):
            if condition():
                return
            await asyncio.sleep(0)
        self.fail("не дождались")


class AskFlowTest(BotCase):
    async def test_answer(self):
        await self.push(update(USER, "Привет"))
        self.assertEqual(self.ask.calls, [([], "Привет")])
        requests = self.session.requests
        typing = [i for i, m in enumerate(requests) if isinstance(m, SendChatAction) and m.action == "typing"]
        answers = [i for i, m in enumerate(requests) if isinstance(m, SendMessage) and m.text == "ответ 1"]
        self.assertTrue(typing)
        self.assertEqual(len(answers), 1)
        self.assertLess(typing[0], answers[0])
        message = requests[answers[0]]
        self.assertEqual(message.chat_id, USER)
        self.assertIsInstance(message.reply_markup, ReplyKeyboardMarkup)
        self.assertEqual(rows(message.reply_markup), [[bot.CLEAR_BUTTON, bot.MENU_BUTTON]])
        self.assertTrue(message.reply_markup.is_persistent)
        self.assertLogged(str(USER), "Привет", "ответ 1")

    async def test_second_question_gets_history(self):
        await self.push(update(USER, "Привет"))
        await self.push(update(USER, "Как дела"))
        history, question = self.ask.calls[1]
        self.assertIsInstance(history, list)
        self.assertEqual(history, [("Привет", "ответ 1")])
        self.assertEqual(question, "Как дела")

    async def test_history_limited(self):
        for i in range(1, 9):
            await self.push(update(USER, f"вопрос {i}"))
        history, question = self.ask.calls[7]
        self.assertEqual(question, "вопрос 8")
        self.assertEqual(len(history), 5)
        # последние 5 пар перед восьмым вопросом, старые первыми
        self.assertEqual(history, [(f"вопрос {i}", f"ответ {i}") for i in range(3, 8)])

    async def test_history_per_user(self):
        await self.push(update(USER, "Привет"))
        await self.push(update(USER2, "Здравствуй"))
        self.assertEqual(self.ask.calls[1], ([], "Здравствуй"))

    async def test_history_per_dispatcher(self):
        await self.push(update(USER, "Привет"))
        await self.push(update(USER, "Снова"), dp=self.make())
        self.assertEqual(self.ask.calls[1], ([], "Снова"))

    async def test_admin_not_in_allowed(self):
        await self.push(update(ADMIN, "Привет"), dp=self.make(allowed={USER}))
        self.assertEqual(self.ask.calls, [([], "Привет")])
        self.assertTrue(any(m.chat_id == ADMIN and m.text == "ответ 1" for m in self.sent()))

    async def test_group_silent(self):
        for sender in (USER, OTHER):
            message = update(sender, "Привет")
            message["message"]["chat"] = GROUP
            await self.push(message)
        press = callback(USER, "clear:yes", 5)
        press["callback_query"]["message"]["chat"] = GROUP
        await self.push(press)
        self.assertEqual(self.ask.calls, [])
        self.assertEqual(self.session.requests, [])
        self.assertEqual(self.log_session.requests, [])

    async def test_no_typing_task_left(self):
        before = asyncio.all_tasks()
        await self.push(update(USER, "Привет"))
        self.ask.error = ai.AIError("m1: HTTP 500")
        await self.push(update(USER, "Упадёт"))
        self.assertEqual(asyncio.all_tasks() - before, set())

    async def test_long_question_log(self):
        await self.push(update(USER, "я" * 5000))
        [text] = self.logs()
        self.assertIn("я" * 1000, text)
        self.assertLessEqual(ai.tg_len(text), 4096)

    async def test_long_answer_log(self):
        self.ask.answer = "б" * 400
        await self.push(update(USER, "Привет"))
        self.assertEqual(self.sent()[-1].text, "б" * 400)
        self.assertLogged("б" * 300 + "…")
        self.assertFalse(any("б" * 301 in text for text in self.logs()))

    async def test_busy(self):
        self.assertEqual(bot.BUSY_TEXT, "Подожди, ещё отвечаю на прошлый вопрос.")
        self.ask.gate = asyncio.Event()
        first = asyncio.create_task(self.push(update(USER, "Первый")))
        await self.until(lambda: self.ask.calls)
        second = update(USER, "Второй")
        await self.push(second)
        self.assertEqual(len(self.ask.calls), 1)
        busy = self.sent()[-1]
        self.assertEqual(busy.text, bot.BUSY_TEXT)
        self.assertEqual(busy.chat_id, USER)
        self.ask.gate.set()
        await first
        self.assertEqual(self.sent()[-1].text, "ответ 1")
        # второй вопрос не встал в очередь и не попал в историю
        self.assertEqual(len(self.ask.calls), 1)
        await self.push(update(USER, "Третий"))
        self.assertEqual(self.ask.calls[1], ([("Первый", "ответ 1")], "Третий"))
        # и входящее, и BUSY отслежены — «Принять» их удаляет
        await self.push(update(USER, bot.CLEAR_BUTTON))
        await self.push(callback(USER, "clear:yes", self.sent_ids(bot.CONFIRM_TEXT)[0]))
        self.assertLessEqual({second["message"]["message_id"], self.sent_ids(bot.BUSY_TEXT)[0]}, self.deleted())


class SendTest(BotCase):
    """send: отправка пользователю с одним повтором при флуд-контроле; ошибки не бросает."""

    async def test_ok(self):
        message = await bot.send(self.tg, USER, "текст", reply_markup=bot.MAIN_KB)
        self.assertEqual(message.message_id, 100)
        [request] = self.sent()
        self.assertEqual((request.chat_id, request.text, request.reply_markup), (USER, "текст", bot.MAIN_KB))

    async def test_retry_once(self):
        self.session.retry_after = 1
        message = await bot.send(self.tg, USER, "текст")
        self.assertEqual(message.text, "текст")
        self.assertEqual(len(self.sent()), 2)

    async def test_retry_only_once(self):
        self.session.retry_after = 2
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(await bot.send(self.tg, USER, "текст"))
        self.assertEqual(len(self.sent()), 2)

    async def test_fail(self):
        self.session.fail_send = True
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(await bot.send(self.tg, USER, "текст"))

    async def test_decode_error(self):
        self.session.decode_error = True
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(await bot.send(self.tg, USER, "текст"))

    async def test_answer_after_flood(self):
        self.session.retry_after = 1
        await self.push(update(USER, "Привет"))
        self.assertEqual([m.text for m in self.sent()], ["ответ 1", "ответ 1"])
        self.assertEqual(self.sent_ids("ответ 1"), [100])
        await self.push(update(USER, "Ещё"))
        self.assertEqual(self.ask.calls[1], ([("Привет", "ответ 1")], "Ещё"))

    async def test_answer_not_sent(self):
        self.session.fail_send = True
        await self.push(update(USER, "Привет"))  # исключение не должно вылететь
        self.assertLogged("Не смог отправить ответ", str(USER), "Привет", "chat not found")
        # неотправленный ответ в историю не попал
        self.session.fail_send = False
        await self.push(update(USER, "Ещё"))
        self.assertEqual(self.ask.calls[1], ([], "Ещё"))

    async def test_answer_not_sent_decode_error(self):
        await self.push(start(USER))
        self.session.decode_error = True  # падает всё: «печатает», удаление меню, ответ
        await self.push(update(USER, "Привет"))
        await self.push(update(USER, bot.MENU_BUTTON))
        self.assertLogged("Не смог отправить ответ", str(USER), "Привет")
        self.session.decode_error = False
        await self.push(update(USER, "Ещё"))
        self.assertEqual(self.ask.calls[1], ([], "Ещё"))

    async def test_all_handlers_survive(self):
        self.session.fail_send = True
        for data in (start(USER), update(USER, bot.MENU_BUTTON), update(USER, bot.CLEAR_BUTTON),
                     update(USER, sticker=STICKER), update(OTHER, "Секрет"), update(USER, "Привет")):
            await self.push(data)  # ни один обработчик не бросает
        self.ask.error = ai.AIError("m1: HTTP 500")
        await self.push(update(USER, "Упадёт"))
        self.assertEqual(self.session.ids, [])
        self.assertGreaterEqual(len(self.sent()), 7)


class MenuTest(BotCase):
    async def test_question_deletes_menu(self):
        await self.push(start(USER))
        menu = self.sent_ids(bot.menu_text(NAME, URL))[0]
        await self.push(update(USER, "Привет"))
        self.assertIn(menu, self.deleted())
        answer = self.sent()[-1]
        self.assertEqual(answer.text, "ответ 1")
        self.assertEqual(answer.reply_markup, bot.MAIN_KB)
        # меню уже нет — второй вопрос ничего не удаляет
        deletes = len(self.deletes())
        await self.push(update(USER, "Как дела"))
        self.assertEqual(len(self.deletes()), deletes)

    async def test_menu_button(self):
        await self.push(update(USER, "Привет"))
        press = update(USER, bot.MENU_BUTTON)
        await self.push(press)
        self.assertIn(press["message"]["message_id"], self.deleted())
        self.assertEqual(self.sent()[-1].text, bot.menu_text(NAME, URL))
        # меню историю не меняет
        await self.push(update(USER, "Как дела"))
        self.assertEqual(self.ask.calls[1], ([("Привет", "ответ 1")], "Как дела"))

    async def test_menu_twice(self):
        await self.push(update(USER, bot.MENU_BUTTON))
        await self.push(update(USER, bot.MENU_BUTTON))
        first, second = self.sent_ids(bot.menu_text(NAME, URL))
        self.assertIn(first, self.deleted())
        self.assertNotIn(second, self.deleted())

    async def test_parallel_start(self):
        # два /start одновременно — в чате остаётся одно меню
        await asyncio.gather(self.push(start(USER)), self.push(start(USER)))
        first, second = self.sent_ids(bot.menu_text(NAME, URL))
        self.assertIn(first, self.deleted())
        self.assertNotIn(second, self.deleted())


class ClearTest(BotCase):
    async def ask_and_clear(self):
        """/start, вопрос, «Очистить сессию». Возвращает id: /start, вопроса, предупреждения."""
        begin, question = start(USER), update(USER, "Привет")
        await self.push(begin)
        await self.push(question)
        await self.push(update(USER, bot.CLEAR_BUTTON))
        warning = self.sent_ids(bot.CONFIRM_TEXT)[0]
        return begin["message"]["message_id"], question["message"]["message_id"], warning

    async def test_empty(self):
        press = update(USER, bot.CLEAR_BUTTON)
        await self.push(press)
        self.assertEqual(self.sent()[-1].text, bot.EMPTY_TEXT)
        self.assertNotIn(bot.CONFIRM_TEXT, [m.text for m in self.sent()])
        self.assertIn(press["message"]["message_id"], self.deleted())

    async def test_confirm(self):
        await self.push(update(USER, "Привет"))
        press = update(USER, bot.CLEAR_BUTTON)
        await self.push(press)
        message = self.sent()[-1]
        self.assertEqual(message.text, bot.CONFIRM_TEXT)
        self.assertIsInstance(message.reply_markup, InlineKeyboardMarkup)
        self.assertEqual(inline(message.reply_markup),
                         [[(bot.ACCEPT_BUTTON, "clear:yes"), (bot.CANCEL_BUTTON, "clear:no")]])
        self.assertEqual(len(self.ask.calls), 1)
        self.assertIn(press["message"]["message_id"], self.deleted())

    async def test_cancel(self):
        await self.push(update(USER, "Привет"))
        await self.push(update(USER, bot.CLEAR_BUTTON))
        warning = self.sent_ids(bot.CONFIRM_TEXT)[0]
        before = len(self.session.requests)
        await self.push(callback(USER, "clear:no", warning))
        self.assertIn(warning, self.deleted())
        # ответ на нажатие — первым, до удаления
        self.assertIsInstance(self.session.requests[before], AnswerCallbackQuery)
        await self.push(update(USER, "Как дела"))
        self.assertEqual(self.ask.calls[1], ([("Привет", "ответ 1")], "Как дела"))

    async def test_accept(self):
        begin, question, warning = await self.ask_and_clear()
        menu = self.sent_ids(bot.menu_text(NAME, URL))[0]
        answer = self.sent_ids("ответ 1")[0]
        logged = len(self.logs())
        before = len(self.session.requests)
        await self.push(callback(USER, "clear:yes", warning))
        # ответ на нажатие — первым, до удалений
        self.assertIsInstance(self.session.requests[before], AnswerCallbackQuery)
        self.assertLessEqual({begin, menu, question, answer, warning}, self.deleted())
        # после удалений: «Сессия очищена» с нижними кнопками, потом меню последним
        requests = self.session.requests
        last_delete = max(i for i, m in enumerate(requests) if isinstance(m, (DeleteMessage, DeleteMessages)))
        cleared = [i for i, m in enumerate(requests) if isinstance(m, SendMessage) and m.text == bot.CLEARED_TEXT]
        self.assertEqual(len(cleared), 1)
        self.assertLess(last_delete, cleared[0])
        self.assertEqual(requests[cleared[0]].reply_markup, bot.MAIN_KB)
        self.assertEqual(self.sent()[-1].text, bot.menu_text(NAME, URL))
        menus = [i for i, m in enumerate(requests) if isinstance(m, SendMessage) and m.text == self.sent()[-1].text]
        self.assertLess(cleared[0], menus[-1])
        self.assertTrue(any(str(USER) in text for text in self.logs()[logged:]), self.logs())
        # история пуста
        await self.push(update(USER, "Заново"))
        self.assertEqual(self.ask.calls[-1], ([], "Заново"))

    async def test_accept_delete_fails(self):
        self.session.fail_delete = True
        _, _, warning = await self.ask_and_clear()
        # исключение не должно вылететь наружу
        await self.push(callback(USER, "clear:yes", warning))
        self.assertIn(bot.CLEARED_TEXT, [m.text for m in self.sent()])
        self.assertEqual(self.sent()[-1].text, bot.menu_text(NAME, URL))

    async def test_other_user_kept(self):
        await self.push(update(USER, "Привет"))
        await self.push(update(USER2, "Здравствуй"))
        await self.push(update(USER, bot.CLEAR_BUTTON))
        await self.push(callback(USER, "clear:yes", self.sent_ids(bot.CONFIRM_TEXT)[0]))
        self.assertTrue(all(m.chat_id == USER for m in self.deletes()))
        await self.push(update(USER2, "Ещё"))
        self.assertEqual(self.ask.calls[-1], ([("Здравствуй", "ответ 2")], "Ещё"))

    async def test_denied_callback(self):
        await self.push(update(USER, "Привет"))
        before, logged = len(self.session.requests), len(self.logs())
        # чужому отвечаем на каждое нажатие (владельца это не спамит), без лога
        await self.push(callback(OTHER, "clear:yes", 500))
        await self.push(callback(OTHER, "clear:yes", 500))
        new = self.session.requests[before:]
        self.assertEqual(len(new), 2)
        for request in new:
            self.assertIsInstance(request, AnswerCallbackQuery)
            self.assertEqual(request.text, bot.DENIED_TEXT)
        self.assertEqual(self.logs()[logged:], [])
        self.assertFalse(any("Очистил" in text for text in self.logs()))
        self.assertEqual(len(self.ask.calls), 1)
        await self.push(update(USER, "Как дела"))
        self.assertEqual(self.ask.calls[1], ([("Привет", "ответ 1")], "Как дела"))

    async def test_empty_by_history(self):
        # «пусто» решается по истории: после одной ошибки Gemini истории нет — «Сессия уже пуста.»
        self.ask.error = ai.AIError("m1: HTTP 500")
        await self.push(update(USER, "Упадёт"))
        await self.push(update(USER, bot.CLEAR_BUTTON))
        self.assertEqual(self.sent()[-1].text, bot.EMPTY_TEXT)

    async def test_empty_after_accept(self):
        # после «Принять» в чате есть «Сессия очищена.» и меню, но история пуста — снова «Сессия уже пуста.»
        _, _, warning = await self.ask_and_clear()
        await self.push(callback(USER, "clear:yes", warning))
        await self.push(update(USER, bot.CLEAR_BUTTON))
        self.assertEqual(self.sent()[-1].text, bot.EMPTY_TEXT)

    async def test_accept_waits_for_answer(self):
        # «Принять», пока Gemini думает: очистка ждёт ответа и удаляет его тоже
        _, _, warning = await self.ask_and_clear()
        self.ask.gate = asyncio.Event()
        question = update(USER, "Ещё")
        asking = asyncio.create_task(self.push(question))
        await self.until(lambda: len(self.ask.calls) == 2)
        clearing = asyncio.create_task(self.push(callback(USER, "clear:yes", warning)))
        for _ in range(50):
            await asyncio.sleep(0)
        self.assertNotIn(bot.CLEARED_TEXT, [m.text for m in self.sent()])
        self.ask.gate.set()
        await asyncio.gather(asking, clearing)
        answer = self.sent_ids("ответ 2")[0]
        self.assertLessEqual({question["message"]["message_id"], answer, warning}, self.deleted())
        texts = [m.text for m in self.sent()]
        self.assertEqual(texts.count(bot.CLEARED_TEXT), 1)
        self.assertLess(texts.index("ответ 2"), texts.index(bot.CLEARED_TEXT))
        self.assertEqual(texts[-1], bot.menu_text(NAME, URL))
        # история очищена уже после ответа
        await self.push(update(USER, "Заново"))
        self.assertEqual(self.ask.calls[-1], ([], "Заново"))

    async def test_double_accept(self):
        _, _, warning = await self.ask_and_clear()
        before, logged = len(self.session.requests), len(self.logs())
        await asyncio.gather(self.push(callback(USER, "clear:yes", warning)),
                             self.push(callback(USER, "clear:yes", warning)))
        new = self.session.requests[before:]
        self.assertEqual([m.text for m in new if isinstance(m, SendMessage)],
                         [bot.CLEARED_TEXT, bot.menu_text(NAME, URL)])
        self.assertEqual(sum(isinstance(m, AnswerCallbackQuery) for m in new), 2)
        self.assertEqual(sum("Очистил" in text for text in self.logs()[logged:]), 1)

    async def test_stale_accept(self):
        # предупреждение, которого бот не помнит (уже очищено или из прошлого запуска): только удалить его
        await self.push(update(USER, "Привет"))
        before, logged = len(self.session.requests), len(self.logs())
        await self.push(callback(USER, "clear:yes", 555))
        new = self.session.requests[before:]
        self.assertEqual([type(m) for m in new], [AnswerCallbackQuery, DeleteMessage])
        self.assertEqual(new[1].message_id, 555)
        self.assertEqual(self.logs()[logged:], [])
        await self.push(update(USER, "Как дела"))
        self.assertEqual(self.ask.calls[1], ([("Привет", "ответ 1")], "Как дела"))

    async def test_accept_answer_fails(self):
        self.session.fail_answer = True
        _, _, warning = await self.ask_and_clear()
        await self.push(callback(USER, "clear:yes", warning))  # исключение не должно вылететь
        self.assertTrue(self.answered())
        self.assertIn(warning, self.deleted())
        self.assertIn(bot.CLEARED_TEXT, [m.text for m in self.sent()])
        self.assertEqual(self.sent()[-1].text, bot.menu_text(NAME, URL))

    async def test_cancel_answer_fails(self):
        self.session.fail_answer = True
        _, _, warning = await self.ask_and_clear()
        await self.push(callback(USER, "clear:no", warning))  # исключение не должно вылететь
        self.assertIn(warning, self.deleted())

    async def test_accept_decode_error(self):
        _, _, warning = await self.ask_and_clear()
        self.session.decode_error = True
        await self.push(callback(USER, "clear:yes", warning))  # исключение не должно вылететь
        self.assertTrue(any(isinstance(m, DeleteMessages) for m in self.deletes()))

    async def test_old_not_deleted(self):
        self.assertEqual(bot.DELETE_AGE, 47 * 3600)
        start_time = 1000.0
        old, new = update(USER, "Привет"), update(USER, "Ещё")
        with patch("bot.now", return_value=start_time):
            await self.push(old)
        with patch("bot.now", return_value=start_time + 2 * 3600):
            await self.push(new)
        with patch("bot.now", return_value=start_time + bot.DELETE_AGE + 60):
            await self.push(update(USER, bot.CLEAR_BUTTON))
            warning = self.sent_ids(bot.CONFIRM_TEXT)[0]
            await self.push(callback(USER, "clear:yes", warning))
        deleted = self.deleted()
        # старше 47 ч — не удаляем (Telegram всё равно не даст), моложе — удаляем
        self.assertNotIn(old["message"]["message_id"], deleted)
        self.assertNotIn(self.sent_ids("ответ 1")[0], deleted)
        self.assertLessEqual({new["message"]["message_id"], self.sent_ids("ответ 2")[0], warning}, deleted)

    async def many(self):
        """/start, 74 вопроса, «Очистить сессию», «Принять»: 150 отслеженных id.
        Возвращает (ожидаемые id, запросы, сделанные после нажатия «Принять»)."""
        begin = start(USER)
        await self.push(begin)
        questions = [update(USER, f"вопрос {i}") for i in range(74)]
        for question in questions:
            await self.push(question)
        await self.push(update(USER, bot.CLEAR_BUTTON))
        warning = self.sent_ids(bot.CONFIRM_TEXT)[0]
        expected = ({begin["message"]["message_id"], warning} | {q["message"]["message_id"] for q in questions}
                    | {i for m, i in self.session.ids if m.text.startswith("ответ ")})
        self.assertEqual(len(expected), 150)
        before = len(self.session.requests)
        await self.push(callback(USER, "clear:yes", warning))
        return expected, self.session.requests[before:]

    async def test_accept_batches(self):
        expected, new = await self.many()
        batches = [m.message_ids for m in new if isinstance(m, DeleteMessages)]
        self.assertEqual([len(b) for b in batches], [100, 50])
        self.assertEqual(set(batches[0]) | set(batches[1]), expected)
        self.assertFalse(any(isinstance(m, DeleteMessage) for m in new))

    async def test_accept_batches_fail(self):
        self.session.fail_delete = True
        expected, new = await self.many()
        self.assertEqual(sum(isinstance(m, DeleteMessages) for m in new), 2)
        singles = [m.message_id for m in new if isinstance(m, DeleteMessage)]
        self.assertEqual(sorted(singles), sorted(expected))


class DeniedTest(BotCase):
    async def test_text(self):
        await self.push(update(OTHER, "Секрет"))
        self.assertEqual(self.ask.calls, [])
        self.assertEqual(self.sent()[-1].text, bot.DENIED_TEXT)
        self.assertEqual(self.sent()[-1].chat_id, OTHER)
        self.assertLogged(str(OTHER), "Секрет")

    async def test_start(self):
        await self.push(start(OTHER))
        self.assertEqual(self.ask.calls, [])
        self.assertEqual(self.sent()[-1].text, bot.DENIED_TEXT)

    async def test_once_per_hour(self):
        self.assertEqual(bot.DENIED_EVERY, 3600)
        await self.push(update(OTHER, "Раз"))
        await self.push(start(OTHER))
        await self.push(update(OTHER, "Три"))
        self.assertEqual([m.text for m in self.sent() if m.chat_id == OTHER], [bot.DENIED_TEXT])
        self.assertEqual(sum(str(OTHER) in text for text in self.logs()), 1)
        self.assertLogged(str(OTHER), "Раз")
        self.assertEqual(self.ask.calls, [])

    async def test_once_per_user(self):
        await self.push(update(OTHER, "Раз"))
        await self.push(update(OTHER2, "Два"))
        self.assertEqual([(m.chat_id, m.text) for m in self.sent()],
                         [(OTHER, bot.DENIED_TEXT), (OTHER2, bot.DENIED_TEXT)])
        self.assertLogged(str(OTHER), "Раз")
        self.assertLogged(str(OTHER2), "Два")


class ErrorTest(BotCase):
    async def test_ai_error(self):
        await self.push(update(USER, "Привет"))
        self.ask.error = ai.AIError("m1: HTTP 429")
        await self.push(update(USER, "Упадёт"))
        self.assertEqual(self.sent()[-1].text, bot.ERROR_TEXT)
        self.assertLogged("m1: HTTP 429")
        # неудачный вопрос в историю не попал
        self.ask.error = None
        await self.push(update(USER, "Ещё раз"))
        self.assertEqual(self.ask.calls[2], ([("Привет", "ответ 1")], "Ещё раз"))

    async def test_ai_error_keyboard(self):
        self.ask.error = ai.AIError("m1: HTTP 429")
        await self.push(update(USER, "Упадёт"))
        message = self.sent()[-1]
        self.assertEqual(message.text, bot.ERROR_TEXT)
        self.assertEqual(message.reply_markup, bot.MAIN_KB)

    async def test_log_bot_broken(self):
        self.log_session.fail = True
        # исключение не должно вылететь наружу
        await self.push(update(USER, "Привет"))
        self.assertEqual(self.sent()[-1].text, "ответ 1")

    async def test_log_bot_decode_error(self):
        self.log_session.decode_error = True
        await self.push(update(USER, "Привет"))  # исключение не должно вылететь
        self.assertEqual(self.sent()[-1].text, "ответ 1")

    async def test_blocked(self):
        self.assertEqual(bot.BLOCKED_TEXT,
                         "Нейросеть отказалась отвечать на этот вопрос — попробуй переформулировать.")
        self.ask.error = ai.BlockedError("m1: blocked SAFETY")
        await self.push(update(USER, "Плохой вопрос"))
        message = self.sent()[-1]
        self.assertEqual(message.text, bot.BLOCKED_TEXT)
        self.assertEqual(message.reply_markup, bot.MAIN_KB)
        self.assertNotIn(bot.ERROR_TEXT, [m.text for m in self.sent()])
        self.assertLogged(str(USER), "Плохой вопрос", "m1: blocked SAFETY")
        # в историю не попало
        self.ask.error = None
        await self.push(update(USER, "Другой"))
        self.assertEqual(self.ask.calls[1], ([], "Другой"))

    async def test_log_redacts_token(self):
        self.ask.error = ai.AIError("https://api.telegram.org/bot123456:SeCrEt-x_1/sendMessage")
        await self.push(update(USER, "Привет"))
        self.assertLogged("https://api.telegram.org/bot<TOKEN>/sendMessage")
        self.assertFalse(any("SeCrEt" in text for text in self.logs()))


class OtherMessagesTest(BotCase):
    async def test_start(self):
        await self.push(start(USER))
        self.assertEqual(self.ask.calls, [])
        message = self.sent()[-1]
        self.assertEqual(message.text, bot.menu_text(NAME, URL))
        self.assertEqual(message.parse_mode, "HTML")
        self.assertIsInstance(message.reply_markup, InlineKeyboardMarkup)
        self.assertEqual(inline(message.reply_markup), [[(bot.GIFT_BUTTON, URL)]])

    async def test_sticker(self):
        await self.push(update(USER, sticker=STICKER))
        self.assertEqual(self.ask.calls, [])
        self.assertEqual(self.sent()[-1].text, bot.ONLY_TEXT)

    async def test_photo(self):
        await self.push(update(USER, photo=PHOTO))
        self.assertEqual(self.ask.calls, [])
        self.assertEqual(self.sent()[-1].text, bot.ONLY_TEXT)


if __name__ == "__main__":
    unittest.main()
