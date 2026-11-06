#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entrypoint for TopicGuard Bot (Forum Topic Moderator & General Chat Router)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.app import TopicGuardBot
from bot.tg_api import TelegramAPI, TelegramError
from core.config import load_config
from core.store import Store

logger = logging.getLogger("topic_guard.main")

# edited_message бот не обрабатывает — не просим Telegram присылать его.
ALLOWED_UPDATES = ["message", "callback_query"]


def load_offset(path: str | Path) -> int:
    """Вернуть последний успешно обработанный update_id (0, если файла нет)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return 0
    if isinstance(raw, dict):
        raw = raw.get("last_update_id", raw.get("update_id", 0))
    try:
        last_id = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(last_id, 0)


def save_offset(path: str | Path, update_id: int) -> None:
    """Атомарно сохранить последний успешно обработанный update_id."""
    target = Path(path)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        payload = json.dumps({"last_update_id": int(update_id)}, ensure_ascii=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, target)
    except (OSError, TypeError, ValueError) as e:
        logger.warning("Cannot save Telegram offset to %s: %s", target, e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def is_stale(update_id: int, last_id: int) -> bool:
    """True, если апдейт уже обработан (идёт не позже сохранённого)."""
    return int(update_id) <= int(last_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="topic-guard-bot",
        description="Telegram Forum Topic Moderator — keeps keys & free-trial info, moves chatter to General.",
    )
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--bot-token", default=None, help="Override Telegram Bot token")
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.bot_token:
        cfg.bot_token = args.bot_token.strip()

    if not cfg.bot_token:
        logger.error("No bot_token configured! Set it in config.json or TELEGRAM_BOT_TOKEN.")
        return 1

    store = Store(
        db_path=cfg.db_path,
        default_auto_guard=cfg.default_auto_guard,
        default_admin_bypass=cfg.default_admin_bypass,
        default_notify_sec=cfg.default_notify_sec,
    )
    api = TelegramAPI(cfg.bot_token)
    bot = TopicGuardBot(api=api, store=store, config=cfg)
    bot.init_bot_info()

    stop_requested = False

    def _sig_handler(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        logger.info("Received signal %s, shutting down...", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    offset_path = Path(cfg.data_dir) / "telegram_offset.json"
    last_update_id = load_offset(offset_path)
    offset: int = last_update_id + 1
    if last_update_id:
        logger.info("Resuming from update_id %s (offset=%s)", last_update_id, offset)
    logger.info("Polling started (poll_timeout=%ss)...", cfg.poll_timeout)

    while not stop_requested:
        try:
            updates = api.get_updates(
                offset=offset,
                timeout=cfg.poll_timeout,
                allowed_updates=ALLOWED_UPDATES,
            )
            for upd in updates or []:
                try:
                    upd_id = int(upd.get("update_id"))
                except (TypeError, ValueError):
                    logger.warning("Skipping update without valid update_id: %r", upd)
                    continue
                offset = max(offset, upd_id + 1)
                if is_stale(upd_id, last_update_id):
                    logger.debug(
                        "Skipping stale update_id %s (last processed %s)",
                        upd_id,
                        last_update_id,
                    )
                    continue
                try:
                    bot.handle_update(upd)
                except Exception as e:
                    logger.exception("Error handling update %s: %s", upd_id, e)
                else:
                    last_update_id = upd_id
                    save_offset(offset_path, upd_id)
        except TelegramError as e:
            if stop_requested:
                break
            logger.warning("Telegram polling error: %s", e)
            time.sleep(3)
        except Exception as e:
            if stop_requested:
                break
            logger.exception("Unexpected loop error: %s", e)
            time.sleep(3)

    store.close()
    logger.info("Stopped cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
