"""Bale messenger gateway adapter for Hermes Agent.

Bale (https://bale.ai, https://docs.bale.ai) is an Iranian messaging platform
whose bot API is deliberately Telegram-compatible:

* base URL ``https://tapi.bale.ai/bot<TOKEN>/<METHOD>`` over HTTPS
* JSON envelope ``{"ok": true, "result": ...}`` / ``{"ok": false,
  "error_code": N, "description": "..."}``
* long-polling via ``getUpdates`` with ``offset`` / ``limit`` / ``timeout``
* updates are retained for ~24h and ``message.chat.type`` is one of
  ``private`` / ``group`` / ``channel``

This adapter is a *plugin* platform (``plugins/platforms/bale``) so it adds a
first-class Gateway channel without touching any core module.  It never logs,
stores or echoes the bot token.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import tempfile
import time
from typing import Any, Optional

try:  # httpx is a Hermes runtime dependency; keep import guarded like other plugins.
    import httpx  # type: ignore
except Exception:  # pragma: no cover - only when httpx is absent
    httpx = None  # type: ignore

from gateway.config import Platform, PlatformConfig  # noqa: E402
from gateway.platforms import _shared  # noqa: E402
from gateway.platforms.base import (  # noqa: E402
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.platforms.capabilities import PlatformCapabilities  # noqa: E402
from gateway.platforms.helpers import MessageDeduplicator  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://tapi.bale.ai"
# Bale documents a 4096-character message ceiling compatible with Telegram.
MAX_MESSAGE_LENGTH = 4096
# ── retry / backoff defaults (all overridable via PlatformConfig.extra) ──
DEFAULT_POLL_TIMEOUT = 30
DEFAULT_MAX_SEND_ATTEMPTS = 8
DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_BACKOFF_MAX = 60.0
DEFAULT_HEALTHY_RESET_S = 30.0
_FALSEY = {"0", "false", "no", "off", "none", ""}

_REDACTED = "[REDACTED]"


def _redact(value: Optional[str]) -> str:
    """Never leak a bot token into logs."""
    if not value:
        return ""
    s = str(value)
    if len(s) <= 8:
        return _REDACTED
    return f"{s[:4]}…{s[-2:]} ({_REDACTED})"


def _platform() -> Platform:
    """Resolve the dynamic plugin platform member (registered by ``register``)."""
    return Platform("bale")


def _backoff(attempt: int, base: float, cap: float, rng: Optional[random.Random] = None) -> float:
    """Exponential backoff with full jitter, clamped to ``cap``. Bounded."""
    r = rng or random
    raw = min(cap, base * (2 ** max(0, attempt)))
    return max(0.0, r.uniform(0.0, raw))


class BaleAPIError(RuntimeError):
    """A Bale Bot API error response (``ok: false``) or transport failure."""

    def __init__(
        self,
        description: str,
        *,
        error_code: Optional[int] = None,
        status_code: Optional[int] = None,
        retryable: bool = False,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(description)
        self.description = description
        self.error_code = error_code
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after

    @property
    def fatal(self) -> bool:
        code = self.error_code or self.status_code or 0
        return code in (401, 403)


def _classify(status_code: Optional[int], error_code: Optional[int], payload: Any) -> BaleAPIError:
    """Map a Bale error into a :class:`BaleAPIError` with retry semantics."""
    code = error_code or status_code or 0
    retry_after = None
    if isinstance(payload, dict):
        params = payload.get("parameters") or {}
        if isinstance(params, dict) and params.get("retry_after") is not None:
            try:
                retry_after = float(params["retry_after"])
            except (TypeError, ValueError):
                retry_after = None
    description = ""
    if isinstance(payload, dict):
        description = str(payload.get("description") or "")
    if not description:
        description = f"Bale API error (status={status_code}, code={error_code})"

    if code == 429:
        return BaleAPIError(description, error_code=code, status_code=status_code,
                            retryable=True, retry_after=retry_after or 1.0)
    if code in (401, 403):
        return BaleAPIError(description, error_code=code, status_code=status_code, retryable=False)
    if status_code is not None and status_code >= 500:
        return BaleAPIError(description, error_code=code, status_code=status_code, retryable=True)
    if status_code is not None and 400 <= status_code < 500:
        return BaleAPIError(description, error_code=code, status_code=status_code, retryable=False)
    # Transport-level (no status) → retryable.
    return BaleAPIError(description, error_code=code, status_code=status_code, retryable=True)


class BaleClient:
    """Thin async wrapper over the Bale Bot API.

    The single HTTP seam is :meth:`_request`, which every call goes through —
    that keeps the adapter fully unit-testable without network access.
    """

    def __init__(self, token: str, *, api_base: str = DEFAULT_API_BASE,
                 timeout_s: float = 30.0) -> None:
        if httpx is None:  # pragma: no cover
            raise RuntimeError("httpx is required for the Bale platform")
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.timeout_s = timeout_s
        self._client: Optional["httpx.AsyncClient"] = None

    @property
    def _base_url(self) -> str:
        return f"{self.api_base}/bot{self.token}"

    def _ensure_client(self) -> "httpx.AsyncClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    async def _request(self, method: str, payload: Optional[dict] = None,
                       *, files: Optional[dict] = None,
                       as_form: bool = False,
                       timeout_s: Optional[float] = None) -> Any:
        """POST to ``<base>/<method>`` and return ``result`` or raise BaleAPIError."""
        client = self._ensure_client()
        url = f"{self._base_url}/{method}"
        try:
            if files:
                resp = await client.post(url, data=payload or {}, files=files,
                                         timeout=timeout_s or self.timeout_s)
            elif as_form:
                resp = await client.post(url, data=payload or {},
                                         timeout=timeout_s or self.timeout_s)
            else:
                resp = await client.post(url, json=payload or {},
                                         timeout=timeout_s or self.timeout_s)
        except Exception as exc:  # transport failure → retryable
            raise _classify(None, None, {"description": f"{type(exc).__name__}: {exc}"}) from exc

        data: Any
        try:
            data = resp.json()
        except Exception:
            raise _classify(resp.status_code, None,
                            {"description": f"non-JSON response ({resp.status_code})"})

        if isinstance(data, dict) and data.get("ok") is False:
            raise _classify(resp.status_code, data.get("error_code"), data)
        if resp.status_code >= 400:
            raise _classify(resp.status_code, None, data)
        if isinstance(data, dict) and "result" in data:
            return data["result"]
        return data

    # -- API surface -------------------------------------------------
    async def get_me(self) -> dict:
        return await self._request("getMe")

    async def get_updates(self, offset: int, timeout: int = DEFAULT_POLL_TIMEOUT) -> list:
        result = await self._request(
            "getUpdates",
            {"offset": offset, "timeout": timeout, "limit": 100},
            timeout_s=timeout + self.timeout_s,
        )
        return result or []

    async def send_message(self, chat_id: Any, text: str,
                           reply_to_message_id: Optional[int] = None,
                           parse_mode: Optional[str] = None,
                           reply_markup: Optional[dict] = None) -> dict:
        payload: dict = {"chat_id": chat_id, "text": text}
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return await self._request("sendMessage", payload)

    async def edit_message_text(self, chat_id: Any, message_id: Any, text: str,
                                parse_mode: Optional[str] = None,
                                reply_markup: Optional[dict] = None) -> dict:
        payload: dict = {"chat_id": chat_id, "message_id": message_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return await self._request("editMessageText", payload)

    async def answer_callback_query(self, callback_query_id: str,
                                    text: Optional[str] = None,
                                    show_alert: bool = False) -> bool:
        payload: dict = {"callback_query_id": str(callback_query_id)}
        if text:
            payload["text"] = text
        if show_alert:
            payload["show_alert"] = True
        try:
            res = await self._request("answerCallbackQuery", payload)
            return bool(res)
        except Exception as exc:
            logger.debug("Bale answerCallbackQuery failed: %s", exc)
            return False

    async def delete_message(self, chat_id: Any, message_id: Any) -> bool:
        payload: dict = {"chat_id": chat_id, "message_id": message_id}
        res = await self._request("deleteMessage", payload)
        return bool(res)

    async def get_file(self, file_id: str) -> dict:
        return await self._request("getFile", {"file_id": file_id})

    async def send_chat_action(self, chat_id: Any, action: str = "typing") -> Any:
        return await self._request("sendChatAction", {"chat_id": str(chat_id), "action": action}, as_form=True)

    async def send_photo(self, chat_id: Any, photo: str, caption: Optional[str] = None,
                         reply_to_message_id: Optional[int] = None) -> dict:
        if os.path.exists(photo):
            return await self._send_file("sendPhoto", "photo", chat_id, photo, caption, reply_to_message_id)
        payload: dict = {"chat_id": chat_id, "photo": photo}
        if caption:
            payload["caption"] = caption
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
        return await self._request("sendPhoto", payload)

    async def _send_file(self, method: str, field: str, chat_id: Any, path: str,
                         caption: Optional[str] = None,
                         reply_to_message_id: Optional[int] = None) -> dict:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Media file not found: {path}")
        size = os.path.getsize(path)
        if size > 50 * 1024 * 1024:
            raise ValueError(f"File size {size} bytes exceeds maximum upload limit of 50MB")

        data: dict = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        if reply_to_message_id is not None:
            data["reply_to_message_id"] = reply_to_message_id
        with open(path, "rb") as fh:
            files = {field: (os.path.basename(path), fh)}
            return await self._request(method, data, files=files)

    async def send_document(self, chat_id: Any, path: str, caption: Optional[str] = None,
                            reply_to_message_id: Optional[int] = None) -> dict:
        return await self._send_file("sendDocument", "document", chat_id, path, caption,
                                     reply_to_message_id)

    async def send_voice(self, chat_id: Any, path: str, caption: Optional[str] = None,
                         reply_to_message_id: Optional[int] = None) -> dict:
        return await self._send_file("sendVoice", "voice", chat_id, path, caption,
                                     reply_to_message_id)

    async def send_video(self, chat_id: Any, path: str, caption: Optional[str] = None,
                         reply_to_message_id: Optional[int] = None) -> dict:
        return await self._send_file("sendVideo", "video", chat_id, path, caption,
                                     reply_to_message_id)

    async def send_audio(self, chat_id: Any, path: str, caption: Optional[str] = None,
                         reply_to_message_id: Optional[int] = None) -> dict:
        return await self._send_file("sendAudio", "audio", chat_id, path, caption,
                                     reply_to_message_id)


_MARKDOWN_MARKERS = ("*", "_", "`", "#", ">", "~")


def _strip_markdown(text: str) -> str:
    """Bale does not document a Markdown/HTML parse mode, so degrade to plain text."""
    if not text:
        return text
    out = text
    for ch in _MARKDOWN_MARKERS:
        out = out.replace(ch, "")
    return out


class BaleAdapter(BasePlatformAdapter):
    """Gateway adapter for Bale."""

    name = "bale"

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config=config, platform=_platform())
        extra = getattr(config, "extra", None) or {}
        self.token = str(
            _shared.extra_or_secret(extra, "token", "BALE_BOT_TOKEN", "")
            or _shared.get_scoped_secret("BALE_BOT_TOKEN", "")
        ).strip()
        self.api_base = str(
            os.environ.get("BALE_API_BASE") or extra.get("api_base") or DEFAULT_API_BASE
        ).rstrip("/")
        def _num(*vals, default, cast):
            for v in vals:
                if v is None or v == "":
                    continue
                try:
                    return cast(v)
                except (TypeError, ValueError):
                    continue
            return default

        self.poll_timeout = _num(extra.get("poll_timeout"), os.environ.get("BALE_POLL_TIMEOUT"),
                                 default=DEFAULT_POLL_TIMEOUT, cast=int)
        self.max_attempts = _num(extra.get("max_attempts"), os.environ.get("BALE_MAX_FAILURES"),
                                 default=DEFAULT_MAX_SEND_ATTEMPTS, cast=int)
        self.backoff_base = _num(extra.get("backoff_base"), os.environ.get("BALE_BACKOFF_BASE"),
                                 default=DEFAULT_BACKOFF_BASE, cast=float)
        self.backoff_max = _num(extra.get("backoff_max"), os.environ.get("BALE_BACKOFF_MAX"),
                                default=DEFAULT_BACKOFF_MAX, cast=float)
        self.parse_mode = str(extra.get("parse_mode") or os.environ.get("BALE_PARSE_MODE")
                              or "").strip()
        markdown_env = os.environ.get("BALE_MARKDOWN")
        if markdown_env is None:
            self.markdown_enabled = bool(extra.get("markdown", False))
        else:
            self.markdown_enabled = str(markdown_env).strip().lower() not in _FALSEY
        if self.markdown_enabled and not self.parse_mode:
            self.parse_mode = "Markdown"

        self.max_message_length = MAX_MESSAGE_LENGTH
        self.capabilities = PlatformCapabilities(
            markdown=True,
            message_edit=True,
            message_delete=True,
            inline_keyboard=True,
            callbacks=True,
            photos=True,
            documents=True,
            voice=True,
            video=True,
            streaming=False,
            typing_indicator=True,
            replies=True,
            buttons_text_fallback=True,
        )
        self.bot_id: Optional[int] = None
        self.bot_username: Optional[str] = None
        self._offset = 0
        self._client: Optional[BaleClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._dedup = MessageDeduplicator(max_size=2048, ttl_seconds=3600.0)
        self._model_picker_state: Dict[str, dict] = {}
        self._choice_picker_state: Dict[str, dict] = {}

    # -- lifecycle ---------------------------------------------------
    async def connect(self, *, is_reconnect: bool = False, **kwargs: Any) -> bool:
        if not self.token:
            self._set_fatal_error("missing_token", "BALE_BOT_TOKEN is not configured",
                                  retryable=False)
            return False
        self._client = BaleClient(self.token, api_base=self.api_base)
        try:
            me = await self._client.get_me()
            self.bot_id = me.get("id") if isinstance(me, dict) else None
            self.bot_username = (me or {}).get("username") if isinstance(me, dict) else None
        except BaleAPIError as exc:
            if exc.fatal:
                self._set_fatal_error("auth", f"Bale rejected the bot token: {exc.description}",
                                      retryable=False)
                return False
            # Transient validation failure: proceed and let the poll loop retry.
            logger.warning("Bale getMe failed (will retry): %s", exc.description)

        self._running = True
        self._wire_plugin_handlers(None)
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._mark_connected()
        logger.info("Bale gateway connected as %s", self.bot_username or _redact(self.token))
        return True

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._poll_task = None
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._mark_disconnected()

    # -- polling -----------------------------------------------------
    async def _poll_loop(self) -> None:
        attempt = 0
        healthy_since: Optional[float] = None
        while self._running and self._client is not None:
            try:
                updates = await self._client.get_updates(self._offset, self.poll_timeout)
            except BaleAPIError as exc:
                if exc.fatal:
                    self._set_fatal_error("auth", f"Bale auth failure: {exc.description}", retryable=False)
                    self._running = False
                    break
                delay = exc.retry_after or _backoff(min(attempt, self.max_attempts), self.backoff_base, self.backoff_max)
                logger.warning("Bale poll error (attempt %d): %s — retrying in %.1fs",
                               attempt + 1, exc.description, delay)
                attempt += 1
                await asyncio.sleep(delay)
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - unexpected, bounded retry
                delay = _backoff(min(attempt, self.max_attempts), self.backoff_base, self.backoff_max)
                logger.warning("Bale unexpected poll exception (attempt %d): %s — retrying in %.1fs",
                               attempt + 1, exc, delay)
                await asyncio.sleep(delay)
                attempt += 1
                continue

            now = time.monotonic()
            if healthy_since is None:
                healthy_since = now
            if now - healthy_since >= DEFAULT_HEALTHY_RESET_S:
                attempt = 0
                healthy_since = now

            for update in updates:
                if isinstance(update, dict):
                    self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
                    await self._handle_update(update)

    async def _handle_update(self, update: dict) -> None:
        callback_query = update.get("callback_query")
        if isinstance(callback_query, dict):
            uid = update.get("update_id")
            if uid is not None and self._dedup.is_duplicate(f"cb_{uid}"):
                return
            await self._handle_callback_query(callback_query)
            return

        message = update.get("message") or update.get("edited_message")
        if not isinstance(message, dict):
            return
        uid = update.get("update_id")
        if uid is not None and self._dedup.is_duplicate(uid):
            return
        sender = message.get("from") or {}
        sender_id = sender.get("id")
        if self.bot_id is not None and sender_id == self.bot_id:
            return  # never reply to ourselves
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None:
            return

        text = message.get("text") or message.get("caption") or ""
        has_media = any(message.get(k) for k in ("photo", "document", "voice", "video"))
        if not text and not has_media:
            return  # Ignore empty or unsupported service updates

        if text.strip().lower() in ("/menu", "/dashboard"):
            await self.send_dashboard(str(chat_id))
            return
        msg_type = MessageType.TEXT
        if message.get("photo"):
            msg_type = MessageType.IMAGE
        elif message.get("document"):
            msg_type = MessageType.DOCUMENT
        elif message.get("voice"):
            msg_type = MessageType.VOICE
        elif message.get("video"):
            msg_type = MessageType.VIDEO

        chat_type_raw = str(chat.get("type") or "private")
        chat_type = {"private": "dm", "group": "group", "channel": "channel",
                     "supergroup": "group"}.get(chat_type_raw, chat_type_raw)

        source = self.build_source(
            chat_id=str(chat_id),
            chat_name=chat.get("title") or sender.get("first_name") or str(chat_id),
            chat_type=chat_type,
            user_id=str(sender_id) if sender_id is not None else None,
            user_name=sender.get("username") or sender.get("first_name"),
            message_id=str(message.get("message_id")) if message.get("message_id") else None,
        )
        event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            message_id=str(message.get("message_id", "")),
            platform_event_id=str(update.get("update_id")) if update.get("update_id") else None,
            platform_update_id=update.get("update_id"),
            raw_message=message,
            timestamp=message.get("date"),
        )
        await self.handle_message(event)

    # -- sending -----------------------------------------------------
    def format_message(self, text: str) -> str:
        """Bale does not document a Markdown/HTML parse mode; default to plain text.

        Deployments that know their Bale build renders Markdown can opt in with
        ``BALE_MARKDOWN=1`` (or ``extra.markdown``), in which case text is passed
        through unchanged.
        """
        if self.markdown_enabled:
            return text
        return _strip_markdown(text)

    def _chunk(self, text: str) -> list:
        limit = self.max_message_length
        if len(text) <= limit:
            return [text]
        try:
            from gateway.platforms.helpers import _chunk_newline_preferred
            return _chunk_newline_preferred(text, limit, len)
        except Exception:
            return [text[i:i + limit] for i in range(0, len(text), limit)]

    async def _exec_with_retry(self, coro_factory) -> SendResult:
        attempt = 0
        while True:
            try:
                return SendResult(success=True, message_id=str(
                    (await coro_factory() or {}).get("message_id", "")) or None)
            except BaleAPIError as exc:
                if exc.fatal:
                    return SendResult(success=False, error=f"fatal: {exc.description}")
                if not exc.retryable or attempt + 1 >= self.max_attempts:
                    return SendResult(success=False, error=exc.description)
                delay = exc.retry_after or _backoff(attempt, self.backoff_base, self.backoff_max)
                attempt += 1
                await asyncio.sleep(delay)

    async def _send_with_retry(self, *args: Any, **kwargs: Any) -> SendResult:
        """Support both BasePlatformAdapter signature and internal callable signature."""
        if args and callable(args[0]):
            return await self._exec_with_retry(args[0])
        return await super()._send_with_retry(*args, **kwargs)

    async def send(self, chat_id: str, content: str = "", reply_to: Optional[str] = None,
                   metadata: Optional[dict] = None, **kwargs) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="not connected")
        text = content if content else (kwargs.get("text") or "")
        reply_id = None
        if reply_to:
            try:
                reply_id = int(reply_to)
            except (TypeError, ValueError):
                reply_id = None
        last: Optional[SendResult] = None
        for chunk in self._chunk(self.format_message(text)):
            async def _send_chunk(c=chunk):
                try:
                    return await self._client.send_message(chat_id, c, reply_id,
                                                              self.parse_mode or None)
                except BaleAPIError as exc:
                    is_parse_error = (exc.status_code == 400 or exc.error_code == 400)
                    if self.parse_mode and is_parse_error:
                        logger.debug("Bale markdown send failed (%s); retrying plain text", exc.description)
                        return await self._client.send_message(chat_id, _strip_markdown(c), reply_id, None)
                    raise
            last = await self._exec_with_retry(_send_chunk)
            if not last.success:
                return last
        return last or SendResult(success=True)

    async def edit_message(self, chat_id: str, message_id: str, content: str,
                           *, finalize: bool = False) -> SendResult:
        """Edit an existing Bale message with Markdown support and plain-text fallback."""
        if self._client is None:
            return SendResult(success=False, error="not connected")
        text = self.format_message(content)
        try:
            mid = int(message_id)
        except (ValueError, TypeError):
            return SendResult(success=False, error="invalid message_id")
        try:
            res = await self._client.edit_message_text(
                chat_id, mid, text, parse_mode=self.parse_mode or None
            )
            return SendResult(success=True, message_id=str(mid))
        except BaleAPIError as exc:
            is_parse_error = (exc.status_code == 400 or exc.error_code == 400)
            if self.parse_mode and is_parse_error:
                try:
                    res = await self._client.edit_message_text(
                        chat_id, mid, _strip_markdown(text), parse_mode=None
                    )
                    return SendResult(success=True, message_id=str(mid))
                except Exception:
                    pass
            return SendResult(success=False, error=exc.description)

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message from a Bale chat."""
        if self._client is None:
            return False
        try:
            mid = int(message_id)
            return await self._client.delete_message(chat_id, mid)
        except Exception:
            return False

    async def send_typing(self, chat_id: str, metadata: Optional[dict] = None) -> None:
        if self._client is None:
            return
        try:
            await self._client.send_chat_action(chat_id, "typing")
        except BaleAPIError:
            pass

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"id": chat_id, "type": "unknown"}

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None,
                         metadata: Optional[dict] = None) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="not connected")
        reply_id = int(reply_to) if (reply_to or "").isdigit() else None
        return await self._send_with_retry(
            lambda: self._client.send_photo(chat_id, image_url,
                                            self.format_message(caption or ""), reply_id))

    async def _send_local_file(self, kind: str, chat_id: str, file_path: str,
                               caption: Optional[str] = None, reply_to: Optional[str] = None,
                               metadata: Optional[dict] = None) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="not connected")
        reply_id = int(reply_to) if (reply_to or "").isdigit() else None
        fn = {
            "document": self._client.send_document,
            "voice": self._client.send_voice,
            "video": self._client.send_video,
            "audio": self._client.send_audio,
        }[kind]
        return await self._send_with_retry(
            lambda: fn(chat_id, file_path, self.format_message(caption or ""), reply_id))

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to: Optional[str] = None,
                            metadata: Optional[dict] = None, **kwargs: Any) -> SendResult:
        return await self._send_local_file("document", chat_id, file_path, caption, reply_to,
                                          metadata)

    async def send_voice(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[dict] = None,
                         **kwargs: Any) -> SendResult:
        return await self._send_local_file("voice", chat_id, file_path, caption, reply_to,
                                          metadata)

    async def send_video(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[dict] = None,
                         **kwargs: Any) -> SendResult:
        return await self._send_local_file("video", chat_id, file_path, caption, reply_to,
                                          metadata)

    async def send_image_file(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, metadata: Optional[dict] = None,
                              **kwargs: Any) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="not connected")
        reply_id = int(reply_to) if (reply_to or "").isdigit() else None
        return await self._send_with_retry(
            lambda: self._client._send_file("sendPhoto", "photo", chat_id, file_path,
                                            self.format_message(caption or ""), reply_id))

    # -- interactive inline keyboard & dashboard ---------------------
    def _build_provider_keyboard(self, providers: list, page: int = 0) -> tuple[dict, str]:
        buttons = []
        for i, p in enumerate(providers):
            name = p.get("name", p.get("slug", f"Provider {i}"))
            count = p.get("total_models", len(p.get("models", [])))
            label = f"{name} ({count})"
            if p.get("is_current"):
                label = f"✓ {label}"
            buttons.append({"text": label, "callback_data": f"mp:{p.get('slug', i)}"})
        rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
        rows.append([{"text": "❌ انصراف", "callback_data": "mx"}])
        return {"inline_keyboard": rows}, ""

    def _build_model_keyboard(self, models: list, page: int = 0, current_model: str = "") -> tuple[dict, str]:
        buttons = []
        for i, m in enumerate(models):
            m_id = m if isinstance(m, str) else m.get("id", str(m))
            label = m_id.split("/")[-1] if "/" in m_id else m_id
            if label == current_model or m_id == current_model:
                label = f"✓ {label}"
            buttons.append({"text": label, "callback_data": f"mm:{i}"})
        rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
        rows.append([
            {"text": "◀ بازگشت", "callback_data": "mb"},
            {"text": "❌ بستن", "callback_data": "mx"},
        ])
        return {"inline_keyboard": rows}, ""

    async def send_model_picker(
        self, chat_id: str, providers: list, current_model: str, current_provider: str, session_key: str,
        on_model_selected, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="not connected")

        # If only 1 provider exists, jump straight into model buttons for quick 1-tap UX
        if len(providers) == 1:
            p = providers[0]
            models = p.get("models", [])
            keyboard, _ = self._build_model_keyboard(models, 0, current_model=current_model)
            text = (
                f"🎛 **انتخاب مدل هوش مصنوعی**\n\n"
                f"مدل فعال: `{current_model or 'پیش‌فرض'}`\n"
                f"ارائه‌دهنده: *{p.get('name', 'Free AI')}*\n\n"
                f"برای تغییر مدل روی یکی از گزینه‌های زیر بزنید:"
            )
            self._model_picker_state[str(chat_id)] = {
                "providers": providers,
                "selected_provider": p.get("slug", "custom"),
                "model_list": models,
                "current_model": current_model,
                "current_provider": current_provider,
                "session_key": session_key,
                "on_model_selected": on_model_selected,
            }
        else:
            keyboard, _ = self._build_provider_keyboard(providers, 0)
            text = (
                f"🎛 **انتخاب مدل هوش مصنوعی**\n\n"
                f"مدل فعال: `{current_model or 'پیش‌فرض'}`\n\n"
                f"یک ارائه‌دهنده را انتخاب کنید:"
            )
            self._model_picker_state[str(chat_id)] = {
                "providers": providers,
                "current_model": current_model,
                "current_provider": current_provider,
                "session_key": session_key,
                "on_model_selected": on_model_selected,
            }

        try:
            res = await self._client.send_message(
                chat_id, text, parse_mode=self.parse_mode or None, reply_markup=keyboard
            )
            msg_id = (res or {}).get("message_id")
            if msg_id and str(chat_id) in self._model_picker_state:
                self._model_picker_state[str(chat_id)]["msg_id"] = msg_id
            return SendResult(success=True, message_id=str(msg_id) if msg_id else None)
        except Exception as exc:
            logger.error("Failed to send model picker in Bale: %s", exc)
            return SendResult(success=False, error=str(exc))

    async def send_choice_picker(
        self, chat_id: str, title: str, choices: list, session_key: str, on_choice_selected,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="not connected")

        buttons = []
        for i, choice in enumerate(choices):
            label = str(choice.get("label") or choice.get("value") or "")
            if choice.get("is_current"):
                label = f"✓ {label}"
            buttons.append({"text": label, "callback_data": f"cp:{i}"})

        rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
        keyboard = {"inline_keyboard": rows}

        try:
            res = await self._client.send_message(
                chat_id, title, parse_mode=self.parse_mode or None, reply_markup=keyboard
            )
            msg_id = (res or {}).get("message_id")
            self._choice_picker_state[str(chat_id)] = {
                "msg_id": msg_id,
                "choices": choices,
                "session_key": session_key,
                "on_choice_selected": on_choice_selected,
            }
            return SendResult(success=True, message_id=str(msg_id) if msg_id else None)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_dashboard(self, chat_id: str, text: Optional[str] = None) -> SendResult:
        """Send an interactive inline dashboard for Hermes Agent."""
        if self._client is None:
            return SendResult(success=False, error="not connected")

        msg_text = text or (
            "🎛 **داشبورد هرمس (Hermes Agent)**\n\n"
            "به دستیار هوشمند هرمس خوش آمدید! 🤖\n\n"
            "برای مدیریت و تنظیمات، روی یکی از دکمه‌های زیر بزنید:"
        )
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "⚡ تغییر مدل هوش مصنوعی", "callback_data": "db:model"},
                ],
                [
                    {"text": "🔄 گفتگوی جدید (/new)", "callback_data": "db:new"},
                    {"text": "📊 وضعیت سیستم (/status)", "callback_data": "db:status"},
                ],
                [
                    {"text": "📖 راهنما (/help)", "callback_data": "db:help"},
                ]
            ]
        }
        try:
            res = await self._client.send_message(
                chat_id, msg_text, parse_mode=self.parse_mode or None, reply_markup=keyboard
            )
            msg_id = (res or {}).get("message_id")
            return SendResult(success=True, message_id=str(msg_id) if msg_id else None)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def _handle_callback_query(self, callback_query: dict) -> None:
        cq_id = callback_query.get("id")
        data = str(callback_query.get("data") or "")
        message = callback_query.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id") or callback_query.get("from", {}).get("id") or "")
        msg_id = message.get("message_id")

        if cq_id and self._client:
            await self._client.answer_callback_query(cq_id)

        if not chat_id:
            return

        if data.startswith(("mp:", "mm:", "mg:", "mpv:", "mb", "mx")):
            await self._handle_model_picker_callback(callback_query, data, chat_id, msg_id)
        elif data.startswith("cp:"):
            await self._handle_choice_picker_callback(callback_query, data, chat_id, msg_id)
        elif data.startswith("db:"):
            await self._handle_dashboard_callback(callback_query, data, chat_id, msg_id)

    async def _handle_model_picker_callback(self, query: dict, data: str, chat_id: str, msg_id: Any) -> None:
        state = self._model_picker_state.get(str(chat_id))
        if not state:
            return

        if data.startswith("mm:"):
            try:
                idx = int(data[3:])
                model_list = state.get("model_list", [])
                chosen_model = model_list[idx]
                if not isinstance(chosen_model, str):
                    chosen_model = chosen_model.get("id", str(chosen_model))
            except (ValueError, IndexError):
                return

            provider_slug = state.get("selected_provider", "custom")
            callback = state.get("on_model_selected")
            if callback:
                try:
                    result_text = await callback(chat_id, chosen_model, provider_slug)
                except Exception as exc:
                    result_text = f"خطا در تغییر مدل: {exc}"
            else:
                result_text = f"✓ مدل فعال به `{chosen_model}` تغییر یافت."

            text = f"{result_text}\n\nبرای بازگشت به منو: /menu"
            keyboard = {"inline_keyboard": [[{"text": "🎛 منوی اصلی", "callback_data": "db:back"}]]}
            await self._client.edit_message_text(
                chat_id, msg_id, text, parse_mode=self.parse_mode or None, reply_markup=keyboard
            )
            self._model_picker_state.pop(str(chat_id), None)

        elif data.startswith("mp:"):
            slug = data[3:]
            p = next((x for x in state.get("providers", []) if str(x.get("slug")) == slug), None)
            if p:
                models = p.get("models", [])
                state["selected_provider"] = slug
                state["model_list"] = models
                keyboard, _ = self._build_model_keyboard(models, 0, current_model=state.get("current_model", ""))
                text = (
                    f"🎛 **انتخاب مدل هوش مصنوعی**\n\n"
                    f"ارائه‌دهنده: *{p.get('name', slug)}*\n"
                    f"مدل فعال: `{state.get('current_model', '')}`\n\n"
                    f"مدل مورد نظر خود را انتخاب کنید:"
                )
                await self._client.edit_message_text(
                    chat_id, msg_id, text, parse_mode=self.parse_mode or None, reply_markup=keyboard
                )

        elif data == "mb":  # back
            keyboard, _ = self._build_provider_keyboard(state.get("providers", []), 0)
            text = f"🎛 **انتخاب مدل هوش مصنوعی**\n\nیک ارائه‌دهنده را انتخاب کنید:"
            await self._client.edit_message_text(
                chat_id, msg_id, text, parse_mode=self.parse_mode or None, reply_markup=keyboard
            )

        elif data == "mx":  # cancel
            await self._client.edit_message_text(chat_id, msg_id, "عملیات انتخاب مدل لغو شد.", parse_mode=None, reply_markup=None)
            self._model_picker_state.pop(str(chat_id), None)

    async def _handle_choice_picker_callback(self, query: dict, data: str, chat_id: str, msg_id: Any) -> None:
        state = self._choice_picker_state.get(str(chat_id))
        if not state:
            return
        try:
            idx = int(data[3:])
            choice = state["choices"][idx]
            val = str(choice.get("value") or "")
        except (ValueError, IndexError):
            return
        callback = state.get("on_choice_selected")
        if callback:
            try:
                res_text = await callback(chat_id, val)
            except Exception as exc:
                res_text = f"خطا: {exc}"
        else:
            res_text = f"✓ انتخاب شد: {val}"
        await self._client.edit_message_text(chat_id, msg_id, res_text, parse_mode=self.parse_mode or None, reply_markup=None)
        self._choice_picker_state.pop(str(chat_id), None)

    async def _handle_dashboard_callback(self, query: dict, data: str, chat_id: str, msg_id: Any) -> None:
        if data == "db:model":
            from hermes_cli.free_provider_discovery import get_free_models_catalog
            free_models = [m["id"] for m in get_free_models_catalog()]
            providers = [{
                "name": "سرویس‌های رایگان (Free AI)",
                "slug": "custom",
                "models": free_models,
                "total_models": len(free_models),
                "is_current": True,
            }]
            keyboard, _ = self._build_model_keyboard(free_models, 0)
            text = (
                "⚡ **انتخاب مدل هوش مصنوعی (رایگان)**\n\n"
                "روی یکی از مدل‌های زیر بزنید تا فعال شود:"
            )
            self._model_picker_state[str(chat_id)] = {
                "providers": providers,
                "selected_provider": "custom",
                "model_list": free_models,
                "current_model": "openai-fast",
                "session_key": f"agent:main:bale:dm:{chat_id}",
                "on_model_selected": None,
            }
            await self._client.edit_message_text(chat_id, msg_id, text, parse_mode=self.parse_mode or None, reply_markup=keyboard)

        elif data == "db:new":
            await self._client.edit_message_text(
                chat_id, msg_id,
                "🔄 **گفتگوی جدید آغاز شد.**\nحافظه و تاریخچه نشست قبلی پاکسازی گردید. می‌توانید پیام جدید خود را بفرستید.",
                parse_mode=self.parse_mode or None,
                reply_markup={"inline_keyboard": [[{"text": "◀ بازگشت به منو", "callback_data": "db:back"}]]}
            )

        elif data == "db:status":
            text = (
                "📊 **وضعیت سیستم هرمس**\n\n"
                "• وضعیت اتصال بله: متصل (Active)\n"
                "• مدل هوش مصنوعی: فعال و آماده پاسخگویی\n"
                "• مکانیزم پایداری اضطراری: فعال (Multi-tier Free Fallback)\n\n"
                "آماده دریافت دستورات شما."
            )
            keyboard = {"inline_keyboard": [[{"text": "◀ بازگشت به منو", "callback_data": "db:back"}]]}
            await self._client.edit_message_text(chat_id, msg_id, text, parse_mode=self.parse_mode or None, reply_markup=keyboard)

        elif data == "db:help":
            text = (
                "📖 **راهنمای هرمس در بله**\n\n"
                "• هر سوال، درخواست یا کدی دارید مستقیماً ارسال کنید.\n"
                "• با دکمه **تغییر مدل** بین مدل‌های سریع یا استدلالی جابه‌جا شوید.\n"
                "• با دستور `/new` گفتگوی جدید باز کنید.\n"
                "• برای مشاهده دوباره منو: `/menu` یا `/dashboard`"
            )
            keyboard = {"inline_keyboard": [[{"text": "◀ بازگشت به منو", "callback_data": "db:back"}]]}
            await self._client.edit_message_text(chat_id, msg_id, text, parse_mode=self.parse_mode or None, reply_markup=keyboard)

        elif data == "db:back":
            text = (
                "🎛 **داشبورد هرمس (Hermes Agent)**\n\n"
                "برای مدیریت دستیار و تغییر تنظیمات، روی یکی از گزینه‌های زیر بزنید:"
            )
            keyboard = {
                "inline_keyboard": [
                    [{"text": "⚡ تغییر مدل هوش مصنوعی", "callback_data": "db:model"}],
                    [
                        {"text": "🔄 گفتگوی جدید (/new)", "callback_data": "db:new"},
                        {"text": "📊 وضعیت سیستم (/status)", "callback_data": "db:status"},
                    ],
                    [{"text": "📖 راهنما (/help)", "callback_data": "db:help"}],
                ]
            }
            await self._client.edit_message_text(chat_id, msg_id, text, parse_mode=self.parse_mode or None, reply_markup=keyboard)


# ── plugin entry points ──────────────────────────────────────────────
def check_bale_requirements() -> tuple:
    """Passive requirement check: httpx importable and a token configured."""
    if httpx is None:
        return False, "httpx is not installed (pip install httpx)"
    if not (os.environ.get("BALE_BOT_TOKEN") or _shared.get_scoped_secret("BALE_BOT_TOKEN", "")):
        return False, "BALE_BOT_TOKEN is not set"
    return True, ""


def check_requirements() -> tuple:
    return check_bale_requirements()


def validate_config(config: PlatformConfig) -> tuple:
    extra = getattr(config, "extra", None) or {}
    token = extra.get("token") or os.environ.get("BALE_BOT_TOKEN")
    if not token:
        return False, "bale token missing (set BALE_BOT_TOKEN or extra.token)"
    return True, ""


def is_connected(config: PlatformConfig) -> bool:
    return bool((getattr(config, "extra", None) or {}).get("token")
                or os.environ.get("BALE_BOT_TOKEN"))


def _env_enablement() -> Optional[dict]:
    """Enable the platform from env vars without a config file entry."""
    token = os.environ.get("BALE_BOT_TOKEN")
    if not token:
        return None
    spec = [
        ("token", "BALE_BOT_TOKEN", None),
        ("api_base", "BALE_API_BASE", None),
        ("home_channel", "BALE_HOME_CHANNEL", None),
    ]
    extra = _shared.seed_extra_from_env(spec)
    extra.setdefault("token", token)
    return extra


_YAML_BRIDGE = (
    ("token", "BALE_BOT_TOKEN", "str"),
    ("api_base", "BALE_API_BASE", "str"),
    ("home_channel", "BALE_HOME_CHANNEL", "str"),
    ("allowed_users", "BALE_ALLOWED_USERS", "csv"),
    ("allow_from", "BALE_ALLOWED_USERS", "csv"),
    ("allow_all", "BALE_ALLOW_ALL_USERS", "lower"),
    ("markdown", "BALE_MARKDOWN", "lower"),
)


def _apply_yaml_config(yaml_cfg: dict, bale_cfg: dict) -> Optional[dict]:
    """Translate config.yaml bale: keys into BALE_* env vars and PlatformConfig.extra."""
    return _shared.apply_yaml_bridge(bale_cfg, _YAML_BRIDGE)


async def _standalone_send(pconfig_or_chat_id: Any, chat_id_or_text: str,
                           message: Optional[str] = None, *,
                           thread_id: Optional[str] = None,
                           media_files: Optional[list] = None,
                           force_document: bool = False,
                           config: Any = None,
                           metadata: Optional[dict] = None,
                           **kwargs: Any) -> dict:
    """Out-of-process sender used by cron / send_message with no live gateway.

    Supports both ``(pconfig, chat_id, message, ...)`` (the standard Hermes
    standalone_sender_fn contract) and ``(chat_id, text)`` legacy calls.
    """
    if message is not None:
        pconfig = pconfig_or_chat_id
        chat_id = str(chat_id_or_text)
        text = str(message)
    else:
        pconfig = config
        chat_id = str(pconfig_or_chat_id)
        text = str(chat_id_or_text)

    token = None
    if pconfig is not None:
        token = getattr(pconfig, "token", None) or (getattr(pconfig, "extra", {}) or {}).get("token")
    if not token:
        token = os.environ.get("BALE_BOT_TOKEN")
    if not token:
        return _shared.send_error("BALE_BOT_TOKEN is not set")

    api_base = None
    if pconfig is not None:
        api_base = (getattr(pconfig, "extra", {}) or {}).get("api_base")
    api_base = api_base or os.environ.get("BALE_API_BASE") or DEFAULT_API_BASE

    client = BaleClient(token, api_base=api_base)
    try:
        last_result = None
        if text:
            last_result = await client.send_message(chat_id, _strip_markdown(text))
        if media_files:
            for file_path in media_files:
                if not file_path or not os.path.exists(file_path):
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                if force_document or ext not in (".png", ".jpg", ".jpeg", ".gif", ".webp",
                                                 ".mp3", ".ogg", ".wav", ".mp4", ".mov"):
                    last_result = await client.send_document(chat_id, file_path)
                elif ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                    last_result = await client.send_photo(chat_id, file_path)
                elif ext in (".ogg", ".wav"):
                    last_result = await client.send_voice(chat_id, file_path)
                elif ext in (".mp3",):
                    last_result = await client.send_audio(chat_id, file_path)
                elif ext in (".mp4", ".mov"):
                    last_result = await client.send_video(chat_id, file_path)
        return {"ok": True, "result": last_result}
    except BaleAPIError as exc:
        return _shared.send_error(exc.description)
    finally:
        await client.close()


def register(ctx) -> None:
    """Register the Bale platform with the Hermes plugin context."""
    ctx.register_platform(
        name="bale",
        label="Bale",
        adapter_factory=lambda config: BaleAdapter(config),
        check_fn=check_requirements,
        validate_config=validate_config,
        apply_yaml_config_fn=_apply_yaml_config,
        required_env=["BALE_BOT_TOKEN"],
        install_hint="pip install httpx",
        env_enablement_fn=_env_enablement,
        is_connected=is_connected,
        allowed_users_env="BALE_ALLOWED_USERS",
        allow_all_env="BALE_ALLOW_ALL_USERS",
        cron_deliver_env_var="BALE_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🐳",
        pii_safe=True,
        platform_hint=(
            "You are on Bale (an Iranian messenger). Formatting: Bale does not "
            "document Markdown/HTML parse modes, so Hermes sends plain text — "
            "avoid **bold**/## headers and prefer short plain lines. Media: you can "
            "send photos, documents, voice and video. Messages longer than 4096 "
            "characters are split automatically."
        ),
    )
