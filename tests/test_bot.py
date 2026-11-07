"""Unit tests for TopicGuard Bot (classifier, store, and General forwarding logic)."""

from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.app import TopicGuardBot
from bot.tg_api import MockTelegramAPI
from core.classifier import classify_message, classify_topic_name
from core.config import Config
from core.store import Store


class TestClassifier(unittest.TestCase):
    def test_keep_api_keys_and_promos(self) -> None:
        samples = [
            {"text": "Ловите ключ OpenAI: sk-proj-1234567890abcdef1234567890"},
            {"text": "AIzaSyD-1234567890abcdef1234567890123"},
            {"text": "Промокод: CURSOR2026FREE"},
            {"text": "FREE-2026-TRIAL-ABCD"},
            {"text": "acc1@gmail.com:SuperSecret123!"},
            {"text": "Держите промик `CLAUDE7DAYS` на неделю"},
        ]
        for msg in samples:
            res = classify_message(msg, mode="smart")
            self.assertTrue(res.keep, f"Expected to keep: {msg['text']} -> {res}")

    def test_keep_free_trial_info_and_links(self) -> None:
        samples = [
            {"text": "Новый способ абуза фри триала Cursor Pro через виртуальную карту:\n1. Регаем аккаунт\n2. Вводим бин 546775"},
            {"text": "Рефка на бесплатную неделю Claude Pro: https://claude.ai/referral/abc123xyz"},
            {"text": "Рабочий БИН для триала ChatGPT Plus: 428837xxxx, зип 10001, проходит с американским IP"},
            {"text": "#инфо Сейчас раздают месяц Perplexity Pro бесплатно через студенческую почту"},
            {"document": {"file_id": "doc1", "file_name": "keys.txt"}, "caption": "Пачка свежих акков"},
        ]
        for msg in samples:
            res = classify_message(msg, mode="smart")
            self.assertTrue(res.keep, f"Expected to keep trial info: {msg} -> {res}")

    def test_move_chatter_questions_and_thanks(self) -> None:
        samples = [
            {"text": "спс забрал"},
            {"text": "первый забрал, от души!"},
            {"text": "не ворк уже"},
            {"text": "все коды убили"},
            {"text": "а как сделать фри триал?"},
            {"text": "где взять бин для курсора?"},
            {"text": "у кого есть рабочий ключ?"},
            {"text": "скиньте под хайд пж"},
            {"text": "работает еще?"},
            {"text": "🔥👍"},
            {"sticker": {"file_id": "st1"}},
            {"voice": {"file_id": "vc1"}},
        ]
        for msg in samples:
            res = classify_message(msg, mode="smart")
            self.assertFalse(res.keep, f"Expected to move chatter: {msg} -> {res}")

    def test_topic_name_classification(self) -> None:
        self.assertEqual(classify_topic_name("General"), "off")
        self.assertEqual(classify_topic_name("Общий чат / Флуд"), "off")
        self.assertEqual(classify_topic_name("Раздача ключей"), "smart")
        self.assertEqual(classify_topic_name("Фри триалы и схемы"), "smart")


class TestTopicGuardBotFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"
        self.store = Store(self.db_path)
        self.cfg = Config(
            bot_token="123:TEST",
            admin_tg_ids=[1001],
            owner_name="Owner",
            owner_username="owner_user",
            owner_tg_ids=[999],
        )
        # Приватка в тестах — по chat_id (титул больше не влияет)
        self.cfg.private_chat_id = -1001999888777
        self.api = MockTelegramAPI(admin_user_ids=[1001])
        self.bot = TopicGuardBot(api=self.api, store=self.store, config=self.cfg)
        self.bot.init_bot_info()
        # Модульная память повторов не должна протекать между тестами
        from core import persona as _persona
        _persona.reset_chat_state()
        # Авто-реакции случайны: в тестах отключаем, иначе ответ уходит
        # анимацией вместо текста и проверки «sent["text"]» падают
        self.bot.media_bank.maybe_reaction = lambda *a, **k: (None, None)
        self.bot.media_bank.wants_reaction_gif = lambda *a, **k: None
        # Фоновые задачи в тестах выполняем синхронно: иначе assertions
        # гоняются с воркерами пула и с закрытой после tearDown БД.
        self.bot._inline_persona = True
        self.bot._submit = lambda fn, *a: fn(*a)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_is_owner_by_id_username_and_stranger(self) -> None:
        self.assertTrue(self.bot._is_owner(999))
        self.assertFalse(self.bot._is_owner(1001))
        self.assertFalse(self.bot._is_owner(4242))
        self.assertTrue(self.bot._is_owner(4242, "owner_user"))
        self.assertTrue(self.bot._is_owner(4242, "@OWNER_USER"))
        self.assertFalse(self.bot._is_owner(4242, "someone_else"))

    def test_keeps_keys_and_moves_chatter_to_general_with_reply_link(self) -> None:
        chat_id = -1001999888777
        topic_id = 42

        # 1. User posts a valid API key in Topic #42 ("Ключи и Триалы")
        key_update = {
            "update_id": 1,
            "message": {
                "message_id": 101,
                "is_topic_message": True,
                "message_thread_id": topic_id,
                "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка Гоя"},
                "from": {"id": 2002, "is_bot": False, "first_name": "Dropper", "username": "dropper_guy"},
                "reply_to_message": {
                    "message_id": topic_id,
                    "forum_topic_created": {"name": "Ключи и Фри Триалы"},
                },
                "text": "Держите свежий ключ: sk-proj-ABCDEF1234567890abcdef123456",
            },
        }
        self.bot.handle_update(key_update)

        # Key message should be kept (not deleted, nothing sent to General)
        self.assertEqual(len(self.api.deleted_messages), 0)
        self.assertEqual(len(self.api.sent_messages), 0)

        # 2. Another user replies to that key post with "спс забрал, а еще будут?"
        chatter_update = {
            "update_id": 2,
            "message": {
                "message_id": 102,
                "is_topic_message": True,
                "message_thread_id": topic_id,
                "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка Гоя"},
                "from": {"id": 3003, "is_bot": False, "first_name": "Vasya", "username": "vasya_goy"},
                "reply_to_message": {
                    "message_id": 101,
                    "text": "Держите свежий ключ: sk-proj-ABCDEF1234567890abcdef123456",
                },
                "text": "спс забрал, а когда еще будут?",
            },
        }
        self.bot.handle_update(chatter_update)

        # Chatter message should be forwarded to General (thread_id=None) and deleted from topic 42
        self.assertEqual(len(self.api.sent_messages), 1)
        sent = self.api.sent_messages[0]
        self.assertEqual(sent["chat_id"], chat_id)
        self.assertIsNone(sent["message_thread_id"])
        self.assertIn("vasya_goy", sent["text"])
        self.assertIn("#Ключи и Фри Триалы", sent["text"])
        self.assertIn("https://t.me/c/1999888777/42/101", sent["text"])
        self.assertIn("спс забрал, а когда еще будут?", sent["text"])

        self.assertEqual(len(self.api.deleted_messages), 1)
        self.assertEqual(self.api.deleted_messages[0]["message_id"], 102)

    def test_custom_general_topic_and_guard_command(self) -> None:
        chat_id = -1001999888777
        general_topic_id = 10
        keys_topic_id = 55

        # Admin sets topic #10 as custom General topic via /setgeneral
        self.bot.handle_update(
            {
                "update_id": 10,
                "message": {
                    "message_id": 200,
                    "is_topic_message": True,
                    "message_thread_id": general_topic_id,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                    "from": {"id": 1001, "is_bot": False, "first_name": "Admin"},
                    "text": "/setgeneral",
                },
            }
        )
        self.assertEqual(self.store.get_chat(chat_id)["general_thread_id"], general_topic_id)

        # Off-topic in topic #55 should now be routed to message_thread_id=10
        self.api.sent_messages.clear()
        self.api.deleted_messages.clear()

        self.bot.handle_update(
            {
                "update_id": 11,
                "message": {
                    "message_id": 201,
                    "is_topic_message": True,
                    "message_thread_id": keys_topic_id,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                    "from": {"id": 4004, "is_bot": False, "first_name": "Petya"},
                    "text": "не ворк уже((",
                },
            }
        )
        self.assertEqual(len(self.api.sent_messages), 1)
        self.assertEqual(self.api.sent_messages[0]["message_thread_id"], general_topic_id)
        self.assertEqual(len(self.api.deleted_messages), 1)
        self.assertEqual(self.api.deleted_messages[0]["message_id"], 201)

    def test_active_dialogue_continuation_in_general(self) -> None:
        chat_id = -1001999888777
        self.bot.config.ai_enabled = True
        self.bot.config.ai_api_key = "test_key"
        self.bot.config.ai_api_url = "http://fake-api"
        self.bot._chat_limiter.per_user_sec = 0.0
        import bot.app
        orig_generate_reply = bot.app.generate_reply
        calls = []
        try:
            def fake_generate_reply(*args, **kwargs):
                calls.append((args, kwargs))
                return "ответ бота"
            bot.app.generate_reply = fake_generate_reply

            # 1. User calls "киря"
            self.bot.handle_update({
                "update_id": 50,
                "message": {
                    "message_id": 501,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                    "from": {"id": 999, "is_bot": False, "first_name": "Owner"},
                    "text": "киря",
                }
            })
            self.assertEqual(len(calls), 1)

            # 2. User sends follow-up without name in active conversation
            self.bot.handle_update({
                "update_id": 51,
                "message": {
                    "message_id": 502,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                    "from": {"id": 999, "is_bot": False, "first_name": "Owner"},
                    "text": "да а у ткбя как дела",
                }
            })
            self.assertEqual(len(calls), 2)
            hist = calls[1][1].get("history")
            self.assertIsNotNone(hist)
            self.assertEqual(len(hist), 2)
            self.assertIn("киря", hist[0]["content"])
            self.assertEqual(hist[1]["content"], "ответ бота")
        finally:
            bot.app.generate_reply = orig_generate_reply

    def test_voice_mode_sends_voice_instead_of_text(self) -> None:
        chat_id = -1001999888777
        self.bot.config.ai_enabled = True
        self.bot.config.ai_api_key = "test_key"
        self.bot.config.ai_api_url = "http://fake-api"
        self.bot._chat_limiter.per_user_sec = 0.0
        self.store.set_setting("tts.voice_replies", "1")
        self.bot._speak = lambda text, voice_id=None: b"\x00" * 4000
        self.api.sent_messages.clear()

        import bot.app
        orig_generate_reply = bot.app.generate_reply
        try:
            bot.app.generate_reply = lambda *a, **k: "ну привет, чё хотел?"
            self.bot.handle_update({
                "update_id": 60,
                "message": {
                    "message_id": 601,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                    "from": {"id": 999, "is_bot": False, "first_name": "Owner"},
                    "text": "киря привет",
                }
            })
        finally:
            bot.app.generate_reply = orig_generate_reply

        self.assertEqual(len(self.api.sent_messages), 1)
        self.assertIn("voice_len", self.api.sent_messages[0])
        self.assertNotIn("text", self.api.sent_messages[0])

    def test_persona_dispatch_runs_in_background_and_keeps_order(self) -> None:
        chat_id = -1001999888777
        self.bot._inline_persona = False
        self.bot.config.ai_enabled = True
        self.bot.config.ai_api_key = "test_key"
        self.bot.config.ai_api_url = "http://fake-api"
        self.bot._chat_limiter.per_user_sec = 0.0
        self.api.sent_messages.clear()
        calls = []

        import bot.app
        orig = bot.app.generate_reply

        def fake(*args, **kwargs):
            calls.append(1)
            return "ответ из фона"

        bot.app.generate_reply = fake
        try:
            for mid, txt in ((901, "киря раз"), (902, "киря два")):
                self.bot.handle_update({
                    "update_id": mid,
                    "message": {
                        "message_id": mid,
                        "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                        "from": {"id": 999, "is_bot": False, "first_name": "Owner"},
                        "text": txt,
                    },
                })
            self.bot._persona_pool.shutdown(wait=True)
        finally:
            bot.app.generate_reply = orig

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.api.sent_messages), 2)
        # порядок ответов сохранён (lock на ветку)
        self.assertEqual(
            [m["reply_to_message_id"] for m in self.api.sent_messages], [901, 902]
        )

    def test_reply_to_message_context_and_memory(self) -> None:
        chat_id = -1001999888777
        self.bot.config.ai_enabled = True
        self.bot.config.ai_api_key = "test_key"
        self.bot.config.ai_api_url = "http://fake-api"
        self.bot._chat_limiter.per_user_sec = 0.0

        import bot.app
        orig_generate_reply = bot.app.generate_reply
        calls = []
        try:
            bot.app.generate_reply = lambda *args, **kwargs: "перевод фразы"

            # 1. User sends Chinese quote unaddressed -> saved in chat_memory, no reply triggered
            self.bot.handle_update({
                "update_id": 55,
                "message": {
                    "message_id": 510,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                    "from": {"id": 999, "is_bot": False, "first_name": "Owner"},
                    "text": "来人，喂公子吃烧饼",
                }
            })
            self.assertEqual(len(self.api.sent_messages), 0)

            # 2. User sends 'киря переведи' in reply to message 510
            self.bot.handle_update({
                "update_id": 56,
                "message": {
                    "message_id": 511,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                    "from": {"id": 999, "is_bot": False, "first_name": "Owner"},
                    "text": "киря переведи",
                    "reply_to_message": {
                        "message_id": 510,
                        "from": {"id": 999, "is_bot": False, "first_name": "Owner"},
                        "text": "来人，喂公子吃烧饼",
                    }
                }
            })
            # Verify bot answered
            self.assertEqual(len(self.api.sent_messages), 1)
            sent = self.api.sent_messages[0]
            self.assertIn("перевод фразы", sent["text"])

            # Verify history stored in SQLite contains the Chinese quote
            mem = self.bot.store.get_memory(chat_id, None, limit=10)
            self.assertTrue(any("来人，喂公子吃烧饼" in m["content"] for m in mem))
        finally:
            bot.app.generate_reply = orig_generate_reply

    def test_auto_mute_on_topic_spam(self) -> None:
        chat_id = -1001999888777
        topic_id = 77
        user_id = 5555

        # 1. First off-topic message -> forwarded to general
        self.bot.handle_update({
            "update_id": 61,
            "message": {
                "message_id": 601,
                "is_topic_message": True,
                "message_thread_id": topic_id,
                "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                "from": {"id": user_id, "is_bot": False, "first_name": "Spammer"},
                "text": "лол 1",
            }
        })
        self.assertEqual(len(self.api.restricted_members), 0)

        # 2. Second off-topic message -> forwarded to general
        self.bot.handle_update({
            "update_id": 62,
            "message": {
                "message_id": 602,
                "is_topic_message": True,
                "message_thread_id": topic_id,
                "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                "from": {"id": user_id, "is_bot": False, "first_name": "Spammer"},
                "text": "лол 2",
            }
        })
        self.assertEqual(len(self.api.restricted_members), 0)

        # 3. Third off-topic message within 12s -> TRIGGERS AUTO-MUTE (10 min)
        self.bot.handle_update({
            "update_id": 63,
            "message": {
                "message_id": 603,
                "is_topic_message": True,
                "message_thread_id": topic_id,
                "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                "from": {"id": user_id, "is_bot": False, "first_name": "Spammer"},
                "text": "лол 3",
            }
        })
        self.assertEqual(len(self.api.restricted_members), 1)
        mute_rec = self.api.restricted_members[0]
        self.assertEqual(mute_rec["user_id"], user_id)
        self.assertIsNotNone(mute_rec["until_date"])

        # Check that mute announcement was posted in General
        last_sent = self.api.sent_messages[-1]
        self.assertIn("отправлен в мут", last_sent["text"])

        # 4. Fourth message from muted user is quietly deleted without any forwarding to general
        sent_count_before = len(self.api.sent_messages)
        self.bot.handle_update({
            "update_id": 64,
            "message": {
                "message_id": 604,
                "is_topic_message": True,
                "message_thread_id": topic_id,
                "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                "from": {"id": user_id, "is_bot": False, "first_name": "Spammer"},
                "text": "я еще могу писать?",
            }
        })
        self.assertEqual(len(self.api.sent_messages), sent_count_before)
        deleted_mids = [d["message_id"] for d in self.api.deleted_messages]
        self.assertIn(604, deleted_mids)

    def test_bot_sends_sticker_when_asked(self) -> None:
        chat_id = -1001999888777
        self.bot.config.ai_enabled = True
        self.bot.config.ai_api_key = "test_key"
        self.bot.config.ai_api_url = "http://fake-api"
        self.bot._chat_limiter.per_user_sec = 0.0
        import bot.app
        orig_generate_reply = bot.app.generate_reply
        try:
            bot.app.generate_reply = lambda *args, **kwargs: "на держи стикер"
            self.bot.handle_update({
                "update_id": 70,
                "message": {
                    "message_id": 701,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                    "from": {"id": 888, "is_bot": False, "first_name": "Owner"},
                    "text": "кирюха скинь стикер",
                }
            })
            # Verify sticker was sent
            sent_stickers = [m for m in self.api.sent_messages if "sticker" in m]
            self.assertTrue(len(sent_stickers) >= 1)
        finally:
            bot.app.generate_reply = orig_generate_reply

    def test_vision_reaction_on_photo(self) -> None:
        chat_id = -1001999888777
        self.bot.config.ai_enabled = True
        self.bot.config.ai_api_key = "test_key"
        self.bot.config.vision_enabled = True
        self.bot._chat_limiter.per_user_sec = 0.0
        self.bot.vision = unittest.mock.MagicMock()
        self.bot.vision.analyze_image.return_value = ("🔥", "огонь артик ня!")

        self.bot.handle_update({
            "update_id": 80,
            "message": {
                "message_id": 801,
                "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                "from": {"id": 777, "is_bot": False, "first_name": "Artist"},
                "photo": [{"file_id": "photo_thumb", "width": 100, "height": 100}, {"file_id": "photo_full", "width": 800, "height": 800}],
                "caption": "киря оцени мой арт",
            }
        })
        import time
        time.sleep(0.1)

        # Verify reaction was set
        self.assertTrue(len(self.api.reactions) >= 1)
        self.assertEqual(self.api.reactions[0]["reaction"], "🔥")

        # Verify comment was sent
        comments = [m for m in self.api.sent_messages if "огонь артик" in m.get("text", "")]
        self.assertTrue(len(comments) >= 1)

    def test_mirror_drop_from_general_to_keys_topic(self) -> None:
        chat_id = -1002000333444
        self.bot.config.private_chat_id = chat_id
        keys_topic_id = 3
        # Register chat & topic
        self.store.ensure_chat(chat_id, title="Ключики от ваших сердец")
        self.store.upsert_topic(chat_id, keys_topic_id, topic_name="Abuse and Keys", mode="smart")

        self.api.copied_messages.clear()
        self.api.deleted_messages.clear()
        self.api.reactions.clear()

        # User drops an API key in General
        self.bot.handle_update({
            "update_id": 90,
            "message": {
                "message_id": 901,
                "is_topic_message": False,
                "chat": {"id": chat_id, "type": "supergroup", "title": "Ключики от ваших сердец"},
                "from": {"id": 1234, "is_bot": False, "first_name": "Dropper"},
                "text": "Ловите рабочий ключ: sk-proj-1234567890abcdef1234567890",
            }
        })

        # Check: drop was mirrored (forwarded or copied) to Abuse & Keys topic
        self.assertEqual(len(self.api.copied_messages), 1)
        self.assertEqual(self.api.copied_messages[0]["message_thread_id"], keys_topic_id)
        self.assertEqual(self.api.copied_messages[0]["source_message_id"], 901)

        # Check: NOT deleted from General!
        self.assertEqual(len(self.api.deleted_messages), 0)

        # Check: reaction set on the drop
        self.assertEqual(len(self.api.reactions), 1)
        self.assertEqual(self.api.reactions[0]["reaction"], "🔥")

    def test_vision_comment_is_not_repeated_on_repeated_gif(self) -> None:
        chat_id = -1001999888777
        tid = 42
        same = "фурр-фурр~ опять этот кот с чипсами на голове, классика жанра"

        class FakeVision:
            """Одинаковая гифка: первые два раза модель выдаёт один и тот же текст."""

            def __init__(self) -> None:
                self.calls = 0
                self.recent_seen: list = []

            def analyze_image(self, image_bytes, caption="", mime="image/jpeg", timeout=18.0, recent=None):
                self.calls += 1
                self.recent_seen.append(list(recent or []))
                if self.calls <= 2:
                    return ("🔥", same)
                return ("😂", "а на второй гифке кот уже уронил клавиатуру прямо в миску")

        self.bot.vision = FakeVision()
        self.api.sent_messages.clear()
        self.api.reactions.clear()

        def send(msg_id: int, user_id: int) -> None:
            self.bot._handle_vision_reaction_and_reply(
                message={}, chat_id=chat_id, tid=tid, mem_tid=tid, msg_id=msg_id,
                file_id="gif1", caption="", mime="image/gif", fallback_file_id=None,
                author_name="Херсов", diag_key=(chat_id, tid),
                user_id=user_id, now=1000.0, is_addressed_or_reply=True,
            )

        send(1, 777)
        self.assertEqual(len(self.api.sent_messages), 1)
        self.assertEqual(self.api.sent_messages[0]["reply_to_message_id"], 1)

        # Другой юзер: rate limit 2 сек на человека не мешает проверке дублей
        send(2, 778)
        # Дубль отброшен, но бот попросил модель ещё раз и выдал новый коммент
        self.assertEqual(len(self.api.sent_messages), 2)
        self.assertIn("уронил клавиатуру", self.api.sent_messages[1]["text"])
        # В промпт попал прошлый комментарий — модель видит, что повторять нельзя
        self.assertTrue(any(same in item for item in self.bot.vision.recent_seen[-1]))
        # Реакция ставится в любом случае
        self.assertEqual([r["message_id"] for r in self.api.reactions], [1, 2])

    def test_vision_sends_no_text_when_model_only_repeats_itself(self) -> None:
        chat_id = -1001999888777
        tid = 42
        same = "фурр-фурр~ опять этот кот с чипсами на голове, классика жанра"

        class StuckVision:
            def analyze_image(self, image_bytes, caption="", mime="image/jpeg", timeout=18.0, recent=None):
                return ("🔥", same)

        self.bot.vision = StuckVision()
        self.api.sent_messages.clear()

        for msg_id in (1, 2):
            self.bot._handle_vision_reaction_and_reply(
                message={}, chat_id=chat_id, tid=tid, mem_tid=tid, msg_id=msg_id,
                file_id="gif1", caption="", mime="image/gif", fallback_file_id=None,
                author_name="Херсов", diag_key=(chat_id, tid),
                user_id=700 + msg_id, now=1000.0, is_addressed_or_reply=True,
            )

        # Первый раз сказал, второй — молчит, чтобы не долбить одно и то же
        self.assertEqual(len(self.api.sent_messages), 1)


    def test_voice_switcher_in_dm(self) -> None:
        dm_id = 999
        admin_id = 1001
        self.bot.muse = None  # без сети: проверяем только переключатель
        self.api.sent_messages.clear()

        self.bot._handle_voice_switcher(dm_id, admin_id, "/voices")
        self.assertEqual(len(self.api.sent_messages), 1)
        msg = self.api.sent_messages[0]
        self.assertIn("Голос Кирюхи", msg["text"])
        kb = msg.get("reply_markup") or {}
        buttons = [b for row in kb.get("inline_keyboard", []) for b in row]
        self.assertTrue(any(b.get("callback_data", "").startswith("tts:set:") for b in buttons))
        self.assertTrue(any(b.get("callback_data") == "tts:mode" for b in buttons))

        # Выбор голоса кнопкой сохраняется в настройках
        self.api.answered_callbacks.clear()
        self.bot._handle_tts_callback("cb1", "tts:set:MBGgaJPW6debJHBqQsSJ", admin_id, dm_id, 1)
        self.assertEqual(self.store.get_setting("tts.voice_id"), "MBGgaJPW6debJHBqQsSJ")
        self.assertEqual(len(self.api.answered_callbacks), 1)

        # Режим голосовых ответов переключается
        target = int(self.bot.config.private_chat_id)
        self.assertFalse(self.bot._voice_mode_on(target))
        self.bot._handle_tts_callback("cb2", "tts:mode", admin_id, dm_id, 1)
        self.assertTrue(self.bot._voice_mode_on(target))
        self.bot._handle_tts_callback("cb3", "tts:mode", admin_id, dm_id, 1)
        self.assertFalse(self.bot._voice_mode_on(target))

    def test_voice_switcher_denies_non_admin(self) -> None:
        self.api.sent_messages.clear()
        self.bot._handle_voice_switcher(555, 999999, "/voices")
        self.assertEqual(len(self.api.sent_messages), 1)
        self.assertIn("только админам", self.api.sent_messages[0]["text"])

    def test_voice_command_sets_named_voice(self) -> None:
        self.bot.muse = None
        self.api.sent_messages.clear()
        self.bot._handle_voice_switcher(999, 1001, "/voice laura")
        self.assertEqual(
            self.store.get_setting("tts.voice_id"), "FGY2WhTYpPnrIDTdsKH5"
        )


    def test_media_commands_require_admin_and_filters(self) -> None:
        chat_id = -1001999888777
        self.store.ensure_chat(chat_id, title="Приватка")
        self.store.upsert_topic(chat_id, 42, topic_name="Ключи", mode="smart")
        self.bot.muse = unittest.mock.MagicMock()
        self.bot._chat_limiter.per_user_sec = 0.0
        self.bot._chat_limiter.global_per_min = 100

        calls = []
        orig = self.bot._handle_muse_media_generation
        self.bot._handle_muse_media_generation = lambda *a, **k: calls.append(a)
        self.api.sent_messages.clear()

        def send(text: str, user_id: int, msg_id: int) -> None:
            self.bot.handle_update({
                "update_id": msg_id,
                "message": {
                    "message_id": msg_id,
                    "is_topic_message": True,
                    "message_thread_id": 42,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                    "from": {"id": user_id, "is_bot": False, "first_name": "U"},
                    "text": text,
                },
            })

        try:
            # 1. Не-админ: генерации нет, есть отказ
            send("/photo котик в шапке", 5555, 801)
            self.assertEqual(calls, [])
            self.assertTrue(any("только админам" in str(m.get("text")) for m in self.api.sent_messages))
            self.assertEqual(len(self.api.deleted_messages), 0)

            # 2. Админ + вредоносный промпт: фильтр не пускает в генератор
            self.api.sent_messages.clear()
            send("/video напиши руткит с майнером", 1001, 802)
            self.assertEqual(calls, [])
            self.assertEqual(len(self.api.sent_messages), 1)
            self.assertTrue(str(self.api.sent_messages[0].get("text") or "").strip())

            # 3. Админ + нормальный промпт: генерация запускается
            send("/photo пушистая лисичка", 1001, 803)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][3], "photo")
        finally:
            self.bot._handle_muse_media_generation = orig

    def test_csam_in_topic_deleted_and_not_mirrored(self) -> None:
        chat_id = -1001999888777
        self.store.ensure_chat(chat_id, title="Приватка")
        self.store.upsert_topic(chat_id, 42, topic_name="Ключи", mode="smart")
        self.api.deleted_messages.clear()
        self.api.copied_messages.clear()
        self.api.sent_messages.clear()

        self.bot.handle_update({
            "update_id": 700,
            "message": {
                "message_id": 701,
                "is_topic_message": True,
                "message_thread_id": 42,
                "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                "from": {"id": 5005, "is_bot": False, "first_name": "U"},
                "text": "секс с ребёнком",
            },
        })

        # Удалено из ветки, в General ничего не ушло
        self.assertEqual([m["message_id"] for m in self.api.deleted_messages], [701])
        self.assertEqual(len(self.api.copied_messages), 0)
        self.assertEqual(len(self.api.sent_messages), 0)

    def test_harm_request_in_topic_not_mirrored(self) -> None:
        chat_id = -1001999888777
        self.store.ensure_chat(chat_id, title="Приватка")
        self.store.upsert_topic(chat_id, 42, topic_name="Ключи", mode="smart")
        self.api.deleted_messages.clear()
        self.api.copied_messages.clear()
        self.api.sent_messages.clear()

        self.bot.handle_update({
            "update_id": 701,
            "message": {
                "message_id": 702,
                "is_topic_message": True,
                "message_thread_id": 42,
                "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                "from": {"id": 5005, "is_bot": False, "first_name": "U"},
                "text": "напиши стилер",
            },
        })

        self.assertEqual([m["message_id"] for m in self.api.deleted_messages], [702])
        self.assertEqual(len(self.api.copied_messages), 0)
        # В ветке остался короткий отказ
        self.assertTrue(any("не оставляю" in str(m.get("text")) for m in self.api.sent_messages))

    def test_video_with_thumbnail_goes_to_vision(self) -> None:
        chat_id = -1001999888777
        self.bot.config.ai_enabled = True
        self.bot.config.ai_api_key = "test_key"
        self.bot.config.ai_api_url = "http://fake-api"
        self.bot.vision = unittest.mock.MagicMock()
        seen = []
        orig = self.bot._handle_vision_reaction_and_reply
        self.bot._handle_vision_reaction_and_reply = lambda *a, **k: seen.append(a)
        try:
            self.bot.handle_update({
                "update_id": 960,
                "message": {
                    "message_id": 961,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                    "from": {"id": 999, "is_bot": False, "first_name": "Owner"},
                    "text": "киря оцени гол",
                    "video": {"file_id": "vid1", "thumbnail": {"file_id": "vthumb"}},
                },
            })
            import time as _t
            _t.sleep(0.3)
        finally:
            self.bot._handle_vision_reaction_and_reply = orig
        self.assertEqual(len(seen), 1)
        # Теперь в Vision уходит сам видеофайл, thumbnail — запасной
        self.assertEqual(seen[0][5], "vid1")
        self.assertEqual(seen[0][8], "vthumb")

    def test_video_without_thumbnail_recorded_in_memory(self) -> None:
        chat_id = -1001999888777
        # Vision выключен: видео без кадра должно хотя бы попасть в память
        self.bot.vision = None
        self.bot.config.ai_enabled = True
        self.bot.config.ai_api_key = "test_key"
        self.bot.config.ai_api_url = "http://fake-api"
        self.bot._chat_limiter.per_user_sec = 0.0
        import bot.app
        orig = bot.app.generate_reply
        try:
            bot.app.generate_reply = lambda *a, **k: "видел гол, красава"
            self.bot.handle_update({
                "update_id": 962,
                "message": {
                    "message_id": 963,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                    "from": {"id": 999, "is_bot": False, "first_name": "Owner"},
                    "text": "киря оцени гол",
                    "video": {"file_id": "vid2", "duration": 12},
                },
            })
        finally:
            bot.app.generate_reply = orig
        rows = self.store.get_memory(chat_id, None, limit=10)
        self.assertTrue(any("[прислал видео]" in r["content"] for r in rows))

    def test_reply_to_gif_with_question_answers_text(self) -> None:
        chat_id = -1001999888777
        self.bot.config.ai_enabled = True
        self.bot.config.ai_api_key = "test_key"
        self.bot.config.ai_api_url = "http://fake-api"
        self.bot._chat_limiter.per_user_sec = 0.0
        self.bot.vision = unittest.mock.MagicMock()
        self.bot.vision.analyze_image.return_value = ("🔥", "кот в наушниках сидит на столе")
        self.api.sent_messages.clear()
        calls = []
        vision_calls = []

        import bot.app
        orig_gen = bot.app.generate_reply
        orig_vis = self.bot._handle_vision_reaction_and_reply
        self.bot._handle_vision_reaction_and_reply = lambda *a, **k: vision_calls.append(a)
        try:
            def fake_gen(*args, **kwargs):
                calls.append(kwargs.get("text") or (args[0] if args else ""))
                return "это адаптер его парсера"
            bot.app.generate_reply = fake_gen
            self.bot.handle_update({
                "update_id": 950,
                "message": {
                    "message_id": 951,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
                    "from": {"id": 999, "is_bot": False, "first_name": "Owner"},
                    "text": "киря что за адаптер у него",
                    "reply_to_message": {
                        "message_id": 900,
                        "from": {"id": 123, "is_bot": False, "first_name": "Th3ryks"},
                        "animation": {"file_id": "gif1", "thumbnail": {"file_id": "thumb1"}},
                    },
                },
            })
        finally:
            bot.app.generate_reply = orig_gen
            self.bot._handle_vision_reaction_and_reply = orig_vis

        # Ответили текстом на вопрос, а не комментарием к гифке
        self.assertEqual(vision_calls, [])
        self.assertEqual(len(calls), 1)
        self.assertIn("что за адаптер", calls[0])
        self.assertIn("кот в наушниках", calls[0])
        self.assertTrue(any("адаптер его парсера" in str(m.get("text")) for m in self.api.sent_messages))

    def test_bot_photo_written_to_memory(self) -> None:
        chat_id = -1001999888777
        self.bot.muse = unittest.mock.MagicMock()
        self.bot.muse.generate_image.return_value = b"\x89PNG" + b"0" * 100
        self.bot.config.muse_video_enabled = True
        self.store.add_memory(chat_id, 42, "user", "Owner", "нарисуй кота")
        self.bot._handle_muse_media_generation(chat_id, 42, 5, "photo", "кот в космосе")
        rows = self.store.get_memory(chat_id, 42, limit=10)
        self.assertTrue(any("[отправил фото: кот в космосе]" in r["content"] for r in rows))

    def test_video_cooldown_skips_attempt(self) -> None:
        chat_id = -1001999888777
        self.bot.config.muse_video_enabled = True
        self.bot.muse = unittest.mock.MagicMock()
        import time as _t
        self.store.set_setting("muse.video_retry_after", str(int(_t.time() + 900)))
        self.api.sent_messages.clear()
        self.bot._handle_muse_media_generation(chat_id, None, 1, "video", "кот")
        self.bot.muse.generate_video.assert_not_called()
        self.assertTrue(any("через" in str(m.get("text")) for m in self.api.sent_messages))

    def test_video_disabled_sends_message_without_generation(self) -> None:
        chat_id = -1001999888777
        self.bot.config.muse_video_enabled = False
        self.bot.muse = unittest.mock.MagicMock()
        self.api.sent_messages.clear()
        self.bot._handle_muse_media_generation(chat_id, None, 1, "video", "кот в космосе")
        self.bot.muse.generate_video.assert_not_called()
        self.assertTrue(any("не отдаёт видео" in str(m.get("text")) for m in self.api.sent_messages))

    def test_dm_voice_reply_and_speed(self) -> None:
        dm_id = 999
        target = int(self.bot.config.private_chat_id)
        self.store.set_setting("tts.voice_replies", "1")
        self.store.set_setting("tts.speed", "1.15")
        self.bot.config.ai_enabled = True
        self.bot.config.ai_api_key = "test_key"
        self.bot.config.ai_api_url = "http://fake-api"
        self.bot._chat_limiter.per_user_sec = 0.0
        self.bot._speak = lambda text, voice_id=None: b"\x00" * 3000
        self.api.sent_messages.clear()

        import bot.app
        orig = bot.app.generate_reply
        try:
            bot.app.generate_reply = lambda *a, **k: "голосом отвечаю"
            self.bot._handle_dm_dialogue(dm_id, dm_id, "привет", "Owner")
        finally:
            bot.app.generate_reply = orig

        self.assertEqual(len(self.api.sent_messages), 1)
        self.assertIn("voice_len", self.api.sent_messages[0])
        self.assertEqual(self.bot._voice_speed(), 1.15)

        # Скорость меняется кнопкой и ограничивается 0.7–1.2
        self.bot._handle_tts_callback("cb", "tts:spd:+0.05", 1001, dm_id, 1)
        self.assertEqual(self.bot._voice_speed(), 1.2)
        self.bot._handle_tts_callback("cb", "tts:spd:+0.05", 1001, dm_id, 1)
        self.assertEqual(self.bot._voice_speed(), 1.2)

    def test_dm_media_generation(self) -> None:
        dm_id = 999
        self.bot.muse = unittest.mock.MagicMock()
        calls = []
        orig = self.bot._handle_muse_media_generation
        self.bot._handle_muse_media_generation = lambda *a, **k: calls.append(a)
        self.bot._chat_limiter.per_user_sec = 0.0
        try:
            self.bot._handle_private_message({
                "message_id": 900,
                "chat": {"id": dm_id, "type": "private"},
                "from": {"id": dm_id, "is_bot": False, "first_name": "Owner"},
                "text": "создай видео будни маратика",
            })
        finally:
            self.bot._handle_muse_media_generation = orig
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][3], "video")

    def test_mirror_not_remembered_when_forward_fails(self) -> None:
        chat_id = -1001999888777
        self.store.ensure_chat(chat_id, title="Приватка")
        self.store.upsert_topic(chat_id, 3, topic_name="Abuse and Keys", mode="smart")
        self.bot.config.private_chat_id = chat_id
        msg = {
            "message_id": 910,
            "chat": {"id": chat_id, "type": "supergroup", "title": "Приватка"},
            "from": {"id": 42, "is_bot": False, "first_name": "D"},
            "text": "Ловите ключ: sk-proj-1234567890abcdef1234567890",
        }
        cfg = self.store.get_chat(chat_id)
        orig_fwd = self.api.forward_message
        self.api.forward_message = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no"))
        orig_copy = self.api.copy_message
        self.api.copy_message = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no"))
        try:
            self.bot._maybe_mirror_drop_from_general(msg, cfg, None)
            # не запомнили как зеркалированное — повтор разрешён
            self.assertNotIn((chat_id, 910), self.bot._mirrored_from_general)
        finally:
            self.api.forward_message = orig_fwd
            self.api.copy_message = orig_copy

    def test_dm_link_delete_requires_target_chat_admin(self) -> None:
        target = -1001999888777
        dm_id = 999
        self.store.ensure_chat(target, title="Приватка")
        self.api.deleted_messages.clear()
        # Владелец — не админ целевого чата в mock, глобальным админом не является
        self.bot._handle_private_message({
            "message_id": 1,
            "chat": {"id": dm_id, "type": "private"},
            "from": {"id": dm_id, "is_bot": False, "first_name": "Owner"},
            "text": f"https://t.me/c/1999888777/555",
        })
        self.assertEqual(self.api.deleted_messages, [])

    def test_sweep_caches_clears_stale_entries(self) -> None:
        chat_id = -1001999888777
        self.bot._active_dialogues[(chat_id, 42)] = (1, 1000.0)
        self.bot._last_media[f"{chat_id}:42"] = ("photo", 1000.0, "кот")
        self.bot._muted_until[(chat_id, 5)] = 1500.0
        self.bot._sweep_caches(now=5000.0)
        self.assertNotIn((chat_id, 42), self.bot._active_dialogues)
        self.assertNotIn(f"{chat_id}:42", self.bot._last_media)
        self.assertNotIn((chat_id, 5), self.bot._muted_until)

    def test_move_failure_keeps_original_message(self) -> None:
        chat_id = -1001999888777
        self.store.ensure_chat(chat_id, title="Приватка")
        chat_cfg = self.store.get_chat(chat_id)
        self.store.upsert_topic(chat_id, 42, topic_name="Ключи", mode="smart")
        topic_cfg = self.store.get_topic(chat_id, 42) or {"thread_id": 42, "topic_name": "Ключи"}
        self.bot.config.ai_enabled = False

        self.api.deleted_messages.clear()
        self.api.sent_messages.clear()
        orig = self.bot._send_to_general

        def boom(*args, **kwargs):
            raise RuntimeError("telegram is down")

        self.bot._send_to_general = boom
        try:
            self.bot._move_message_to_general(
                {"message_id": 777, "text": "оффтоп в ветке", "from": {"id": 5, "first_name": "U"}},
                chat_cfg,
                topic_cfg,
                "offtopic",
            )
        finally:
            self.bot._send_to_general = orig

        # Оригинал не удалён, статистика не выросла
        self.assertEqual(len(self.api.deleted_messages), 0)
        self.assertEqual(self.store.get_chat(chat_id)["moved_count"], chat_cfg["moved_count"])
        # В ветку ушло предупреждение, что перенос не удался
        self.assertTrue(any("не смог перенести" in str(m.get("text")) for m in self.api.sent_messages))

    def test_move_success_deletes_original(self) -> None:
        chat_id = -1001999888777
        self.store.ensure_chat(chat_id, title="Приватка")
        chat_cfg = self.store.get_chat(chat_id)
        self.store.upsert_topic(chat_id, 42, topic_name="Ключи", mode="smart")
        topic_cfg = self.store.get_topic(chat_id, 42) or {"thread_id": 42, "topic_name": "Ключи"}
        self.bot.config.ai_enabled = False
        self.api.deleted_messages.clear()

        self.bot._move_message_to_general(
            {"message_id": 778, "text": "оффтоп в ветке", "from": {"id": 5, "first_name": "U"}},
            chat_cfg,
            topic_cfg,
            "offtopic",
        )
        self.assertEqual(len(self.api.deleted_messages), 1)
        self.assertEqual(self.api.deleted_messages[0]["message_id"], 778)
        self.assertEqual(self.store.get_chat(chat_id)["moved_count"], chat_cfg["moved_count"] + 1)


if __name__ == "__main__":
    unittest.main()
