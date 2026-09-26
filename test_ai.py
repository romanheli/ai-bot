import asyncio
import unittest

import aiohttp

import ai
from testutil import FakeGemini, gemini_ok

KEY = "SECRET-KEY"
URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"


class TgLenTest(unittest.TestCase):
    def test_ascii(self):
        self.assertEqual(ai.tg_len("abc"), 3)

    def test_emoji_is_two(self):
        self.assertEqual(ai.tg_len("😀"), 2)

    def test_cyrillic_and_empty(self):
        self.assertEqual(ai.tg_len("привет"), 6)
        self.assertEqual(ai.tg_len(""), 0)

    def test_limit(self):
        self.assertEqual(ai.TG_LIMIT, 4096)


class CleanTest(unittest.TestCase):
    def test_bold(self):
        self.assertEqual(ai.clean("**жирный**"), "жирный")
        self.assertEqual(ai.clean("это **важно** и **нужно**"), "это важно и нужно")

    def test_headers(self):
        self.assertEqual(ai.clean("## Заголовок"), "Заголовок")
        self.assertEqual(ai.clean("# Один"), "Один")
        self.assertEqual(ai.clean("###### Шесть"), "Шесть")
        self.assertEqual(ai.clean("Текст\n### Раздел\nещё"), "Текст\nРаздел\nещё")

    def test_hash_without_space_untouched(self):
        self.assertEqual(ai.clean("#тег"), "#тег")

    def test_list_items(self):
        self.assertEqual(ai.clean("* пункт"), "• пункт")
        self.assertEqual(ai.clean("- пункт"), "• пункт")
        self.assertEqual(ai.clean("Список:\n* раз\n- два"), "Список:\n• раз\n• два")

    def test_nested_item_keeps_indent(self):
        self.assertEqual(ai.clean("Список:\n  - вложенный\n    * глубже"), "Список:\n  • вложенный\n    • глубже")

    def test_untouched(self):
        for text in ("__init__", "a * b", "-5", "x - y", "итого: -5 градусов"):
            with self.subTest(text=text):
                self.assertEqual(ai.clean(text), text)

    def test_strip(self):
        self.assertEqual(ai.clean("\n  текст  \n\n"), "текст")


class FitTest(unittest.TestCase):
    def test_short_unchanged(self):
        self.assertEqual(ai.fit("короткий текст"), "короткий текст")

    def test_exactly_limit_unchanged(self):
        text = "а" * ai.TG_LIMIT
        self.assertEqual(ai.fit(text), text)

    def test_long_plain(self):
        result = ai.fit("а" * 5000)
        self.assertLessEqual(ai.tg_len(result), 4096)
        self.assertTrue(result.endswith("…"))

    def test_long_emoji(self):
        # 3000 символов, но 6000 единиц UTF-16
        result = ai.fit("😀" * 3000)
        self.assertLessEqual(ai.tg_len(result), 4096)
        self.assertTrue(result.endswith("…"))

    def test_cuts_by_paragraph(self):
        paras = [f"{i}" + "б" * 999 for i in range(8)]
        result = ai.fit("\n".join(paras))
        self.assertLessEqual(ai.tg_len(result), 4096)
        self.assertTrue(result.endswith("…"))
        # до «…» — целые абзацы, последний не оборван
        body = result[:-1].rstrip()
        self.assertIn(body, ["\n".join(paras[:k]) for k in range(1, len(paras))])

    def test_cuts_by_sentence(self):
        # переводов строки нет — режем по ". ", точка остаётся
        result = ai.fit("Это предложение. " * 300)
        self.assertLessEqual(ai.tg_len(result), 4096)
        self.assertTrue(result.endswith(".…"))

    def test_custom_limit(self):
        result = ai.fit("а" * 10, limit=5)
        self.assertLessEqual(ai.tg_len(result), 5)
        self.assertTrue(result.endswith("…"))


class BuildBodyTest(unittest.TestCase):
    def test_with_history(self):
        body = ai.build_body([("q1", "a1"), ("q2", "a2")], "q3")
        self.assertEqual(body, {
            "systemInstruction": {"parts": [{"text": ai.SYSTEM_PROMPT}]},
            "contents": [
                {"role": "user", "parts": [{"text": "q1"}]},
                {"role": "model", "parts": [{"text": "a1"}]},
                {"role": "user", "parts": [{"text": "q2"}]},
                {"role": "model", "parts": [{"text": "a2"}]},
                {"role": "user", "parts": [{"text": "q3"}]},
            ],
        })

    def test_empty_history(self):
        body = ai.build_body([], "вопрос")
        self.assertEqual(body["contents"], [{"role": "user", "parts": [{"text": "вопрос"}]}])
        self.assertEqual(body["systemInstruction"], {"parts": [{"text": ai.SYSTEM_PROMPT}]})


class ExtractTextTest(unittest.TestCase):
    def test_single(self):
        self.assertEqual(ai.extract_text(gemini_ok("ответ")), "ответ")

    def test_parts_joined(self):
        data = {"candidates": [{"content": {"parts": [{"text": "раз "}, {"text": "два"}]}}]}
        self.assertEqual(ai.extract_text(data), "раз два")

    def test_thoughts_skipped(self):
        data = {"candidates": [{"content": {"parts": [{"text": "думаю…", "thought": True}, {"text": "ответ"}]}}]}
        self.assertEqual(ai.extract_text(data), "ответ")

    def test_none(self):
        cases = [
            {},
            {"candidates": []},
            {"candidates": [{"finishReason": "SAFETY"}]},
            {"candidates": [{"content": {"parts": []}}]},
            {"candidates": [{"content": {"parts": [{"text": ""}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "думаю", "thought": True}]}}]},
        ]
        for data in cases:
            with self.subTest(data=data):
                self.assertIsNone(ai.extract_text(data))


class AskTest(unittest.IsolatedAsyncioTestCase):
    MODELS = ["m1", "m2"]

    async def ask(self, routes, history=(), question="вопрос"):
        self.gemini = FakeGemini(routes)
        return await ai.ask(self.gemini, KEY, self.MODELS, list(history), question)

    async def test_first_model(self):
        result = await self.ask({"m1": (200, gemini_ok("ответ")), "m2": (200, gemini_ok("другой"))})
        self.assertEqual(result, ("ответ", "m1"))
        self.assertEqual(self.gemini.models(), ["m1"])

    async def test_429_goes_to_next(self):
        result = await self.ask({"m1": (429, {}), "m2": (200, gemini_ok("ответ"))})
        self.assertEqual(result, ("ответ", "m2"))
        self.assertEqual(self.gemini.models(), ["m1", "m2"])

    async def test_retry_statuses_go_to_next(self):
        for status in (404, 429, 500, 503):
            with self.subTest(status=status):
                result = await self.ask({"m1": (status, {}), "m2": (200, gemini_ok("ответ"))})
                self.assertEqual(result, ("ответ", "m2"))

    async def test_empty_goes_to_next(self):
        result = await self.ask({"m1": (200, {"candidates": []}), "m2": (200, gemini_ok("ответ"))})
        self.assertEqual(result, ("ответ", "m2"))

    async def test_timeout_goes_to_next(self):
        result = await self.ask({"m1": asyncio.TimeoutError(), "m2": (200, gemini_ok("ответ"))})
        self.assertEqual(result, ("ответ", "m2"))

    async def test_connection_goes_to_next(self):
        result = await self.ask({"m1": aiohttp.ClientConnectionError(), "m2": (200, gemini_ok("ответ"))})
        self.assertEqual(result, ("ответ", "m2"))

    async def test_fatal_status_stops(self):
        for status in (400, 403):
            with self.subTest(status=status):
                with self.assertRaises(ai.AIError) as err:
                    await self.ask({"m1": (status, {}), "m2": (200, gemini_ok("ответ"))})
                self.assertIn(f"m1: HTTP {status}", str(err.exception))
                self.assertEqual(self.gemini.models(), ["m1"])

    async def test_all_failed(self):
        with self.assertRaises(ai.AIError) as err:
            await self.ask({"m1": (429, {}), "m2": asyncio.TimeoutError()})
        self.assertIn("m1: HTTP 429; m2: timeout", str(err.exception))

    async def test_all_failed_messages(self):
        with self.assertRaises(ai.AIError) as err:
            await self.ask({"m1": (200, {}), "m2": aiohttp.ClientConnectionError()})
        self.assertIn("m1: empty", str(err.exception))
        self.assertIn("m2: connection", str(err.exception))

    async def test_key_only_in_header(self):
        await self.ask({"m1": (200, gemini_ok("ответ"))})
        url, _, headers = self.gemini.calls[0]
        self.assertEqual(url, URL.format("m1"))
        self.assertNotIn(KEY, url)
        self.assertEqual(headers["x-goog-api-key"], KEY)

    async def test_body(self):
        history = [("q1", "a1")]
        await self.ask({"m1": (200, gemini_ok("ответ"))}, history, "q2")
        self.assertEqual(self.gemini.calls[0][1], ai.build_body(history, "q2"))

    async def test_answer_cleaned(self):
        result = await self.ask({"m1": (200, gemini_ok("**x**"))})
        self.assertEqual(result, ("x", "m1"))

    async def test_answer_fitted(self):
        text, _ = await self.ask({"m1": (200, gemini_ok("а" * 5000))})
        self.assertLessEqual(ai.tg_len(text), ai.TG_LIMIT)
        self.assertTrue(text.endswith("…"))


if __name__ == "__main__":
    unittest.main()
