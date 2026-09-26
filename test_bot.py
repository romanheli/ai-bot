import asyncio
import unittest
from datetime import datetime, timedelta, timezone

from aiogram import Bot
from aiogram.methods import AnswerCallbackQuery, DeleteMessage, DeleteMessages, SendChatAction, SendMessage
from aiogram.types import InlineKeyboardMarkup, ReplyKeyboardMarkup, Update

import ai
import bot
from testutil import (ADMIN, NAME, OTHER, PHOTO, STICKER, URL, USER, USER2, FakeAsk, MockedSession, callback,
                      reset_ids, rows, start, update)


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
        await (dp or self.dp).feed_update(self.tg, Update.model_validate(data, context={"bot": self.tg}))
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
        self.assertEqual(len(history), bot.HISTORY_SIZE)
        # последние пары перед восьмым вопросом, старые первыми
        self.assertEqual(history, [(f"вопрос {i}", f"ответ {i}") for i in range(8 - bot.HISTORY_SIZE, 8)])

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
        await self.push(callback(USER, "clear:no", warning))
        self.assertIn(warning, self.deleted())
        self.assertTrue(self.answered())
        await self.push(update(USER, "Как дела"))
        self.assertEqual(self.ask.calls[1], ([("Привет", "ответ 1")], "Как дела"))

    async def test_accept(self):
        begin, question, warning = await self.ask_and_clear()
        menu = self.sent_ids(bot.menu_text(NAME, URL))[0]
        answer = self.sent_ids("ответ 1")[0]
        logged = len(self.logs())
        await self.push(callback(USER, "clear:yes", warning))
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
        self.assertTrue(self.answered())
        self.assertTrue(any(str(USER) in text for text in self.logs()[logged:]), self.logs())
        # история пуста
        await self.push(update(USER, bot.CLEAR_BUTTON))
        self.assertEqual(self.sent()[-1].text, bot.EMPTY_TEXT)
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
        await self.push(callback(OTHER, "clear:yes", 500))
        self.assertEqual(len(self.ask.calls), 1)
        self.assertEqual(self.deletes(), [])
        self.assertTrue(self.answered())
        await self.push(update(USER, "Как дела"))
        self.assertEqual(self.ask.calls[1], ([("Привет", "ответ 1")], "Как дела"))


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
