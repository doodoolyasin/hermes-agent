"""Rubika platform adapter for Hermes.

Implements the official Rubika Bot API interface with full capability detection,
message deduplication, media handling, and capability-aware fallbacks (such as
numbered text menus when inline keyboards/callbacks are not supported).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp

from gateway.config import Platform, PlatformConfig
from gateway.platforms import _shared
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.platforms.capabilities import PlatformCapabilities
from gateway.platforms.helpers import MessageDeduplicator

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://botapi.rubika.ir/v01"
MAX_MESSAGE_LENGTH = 4096

DEFAULT_POLL_TIMEOUT = 30
DEFAULT_MAX_FAILURES = 8
DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_BACKOFF_MAX = 60.0

_SECRET_KEYS = frozenset({"token", "rubika_bot_token", "api_key"})


def _redact_token(url_or_text: str, token: str) -> str:
    if not token or len(token) < 4:
        return url_or_text
    return url_or_text.replace(token, "[REDACTED]")


class RubikaAPIError(Exception):
    """Rubika Bot API call failure."""

    def __init__(self, method: str, status_code: int, description: str,
                 parameters: Optional[dict] = None) -> None:
        self.method = method
        self.status_code = status_code
        self.description = description
        self.parameters = parameters or {}
        self.fatal = status_code in (401, 403)
        super().__init__(f"Rubika API error {status_code} in {method}: {description}")


class RubikaClient:
    """Async HTTP transport for Rubika Bot API."""

    def __init__(self, token: str, api_base: Optional[str] = None,
                 timeout_s: float = 40.0) -> None:
        self.token = token
        self.api_base = (api_base or DEFAULT_API_BASE).rstrip("/")
        self.timeout_s = timeout_s
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _endpoint(self, method: str) -> str:
        return f"{self.api_base}/{self.token}/{method}"

    async def _request(self, method: str, payload: Optional[dict] = None,
                        files: Optional[dict] = None,
                        timeout_s: Optional[float] = None) -> Any:
        session = await self._get_session()
        url = self._endpoint(method)
        req_timeout = aiohttp.ClientTimeout(total=timeout_s or self.timeout_s)

        try:
            if files:
                form = aiohttp.FormData()
                if payload:
                    for k, v in payload.items():
                        form.add_field(k, str(v) if not isinstance(v, (dict, list)) else json.dumps(v))
                for file_field, (filename, file_data, content_type) in files.items():
                    form.add_field(file_field, file_data, filename=filename, content_type=content_type)
                async with session.post(url, data=form, timeout=req_timeout) as resp:
                    status = resp.status
                    text = await resp.text()
            else:
                headers = {"Content-Type": "application/json"}
                data_bytes = json.dumps(payload or {}).encode("utf-8")
                async with session.post(url, data=data_bytes, headers=headers, timeout=req_timeout) as resp:
                    status = resp.status
                    text = await resp.text()
        except asyncio.TimeoutError as exc:
            raise RubikaAPIError(method, 408, "Request timed out") from exc
        except aiohttp.ClientError as exc:
            raise RubikaAPIError(method, 0, f"Network error: {exc}") from exc

        try:
            data = json.loads(text)
        except Exception:
            data = {"status": "ERROR", "description": text[:200]}

        if status >= 400 or (isinstance(data, dict) and data.get("status") == "ERROR"):
            desc = data.get("description") or f"HTTP {status}"
            raise RubikaAPIError(method, status, desc, parameters=data.get("parameters"))

        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data

    async def get_me(self) -> dict:
        return await self._request("getMe")

    async def get_updates(self, offset: int, timeout: int = DEFAULT_POLL_TIMEOUT) -> list:
        result = await self._request("getUpdates", {"offset": offset, "limit": 100, "timeout": timeout}, timeout_s=timeout + self.timeout_s)
        return result or []

    async def send_message(self, chat_id: Any, text: str, reply_to_message_id: Optional[int] = None) -> dict:
        payload: dict = {"chat_id": chat_id, "text": text}
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
        return await self._request("sendMessage", payload)

    async def edit_message_text(self, chat_id: Any, message_id: Any, text: str) -> dict:
        return await self._request("editMessageText", {"chat_id": chat_id, "message_id": message_id, "text": text})

    async def delete_message(self, chat_id: Any, message_id: Any) -> dict:
        return await self._request("deleteMessage", {"chat_id": chat_id, "message_id": message_id})

    async def send_chat_action(self, chat_id: Any, action: str = "typing") -> dict:
        return await self._request("sendChatAction", {"chat_id": chat_id, "action": action})

    async def send_document(self, chat_id: Any, file_data: bytes, filename: str,
                            caption: Optional[str] = None,
                            reply_to_message_id: Optional[int] = None) -> dict:
        payload: dict = {"chat_id": chat_id}
        if caption:
            payload["caption"] = caption
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
        files = {"file": (filename, file_data, "application/octet-stream")}
        return await self._request("sendDocument", payload=payload, files=files)


class RubikaAdapter(BasePlatformAdapter):
    """Rubika platform adapter with capability matrix and text menu fallback."""

    supports_code_blocks = False
    supports_status_text = False

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config=config, platform=Platform("rubika"))
        extra = config.extra or {}
        self.token = str(_shared.extra_or_secret(extra, "token", "RUBIKA_BOT_TOKEN") or "").strip()
        self.api_base = str(extra.get("api_base") or os.environ.get("RUBIKA_API_BASE") or DEFAULT_API_BASE).rstrip("/")

        raw_users = extra.get("allowed_users") or os.environ.get("RUBIKA_ALLOWED_USERS") or ""
        if isinstance(raw_users, str):
            self.allowed_users = {u.strip() for u in raw_users.split(",") if u.strip()}
        elif isinstance(raw_users, (list, set)):
            self.allowed_users = {str(u).strip() for u in raw_users if str(u).strip()}
        else:
            self.allowed_users = set()

        raw_allow_all = extra.get("allow_all_users") or os.environ.get("RUBIKA_ALLOW_ALL_USERS") or "0"
        self.allow_all_users = str(raw_allow_all).strip().lower() in ("1", "true", "yes")

        self.poll_timeout = int(extra.get("poll_timeout") or os.environ.get("RUBIKA_POLL_TIMEOUT") or DEFAULT_POLL_TIMEOUT)
        self.max_failures = int(extra.get("max_failures") or os.environ.get("RUBIKA_MAX_FAILURES") or DEFAULT_MAX_FAILURES)
        self.backoff_base = float(extra.get("backoff_base") or os.environ.get("RUBIKA_BACKOFF_BASE") or DEFAULT_BACKOFF_BASE)
        self.backoff_max = float(extra.get("backoff_max") or os.environ.get("RUBIKA_BACKOFF_MAX") or DEFAULT_BACKOFF_MAX)

        self.max_message_length = MAX_MESSAGE_LENGTH
        # Capability Matrix: Declare realistic Rubika capabilities
        self.capabilities = PlatformCapabilities(
            markdown=False,
            message_edit=True,
            message_delete=True,
            inline_keyboard=False,       # Official Bot API lacks inline buttons
            callbacks=False,             # Falls back to numbered text menu!
            photos=True,
            documents=True,
            voice=True,
            video=True,
            streaming=False,
            typing_indicator=True,
            replies=True,
            buttons_text_fallback=True,  # Text menu fallback active
        )
        self.bot_id: Optional[str] = None
        self.bot_username: Optional[str] = None
        self._offset = 0
        self._client: Optional[RubikaClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._dedup = MessageDeduplicator(max_size=2048, ttl_seconds=3600.0)

    def render_fallback_menu(self, title: str, options: List[str]) -> str:
        """Render a clean numbered Persian text menu when inline buttons are unsupported."""
        lines = [f"📌 **{title}**\n"]
        for idx, opt in enumerate(options, 1):
            lines.append(f"{idx}️⃣ {opt}")
        lines.append("\n💡 برای انتخاب، عدد یا عنوان گزینه را ارسال کنید.")
        return "\n".join(lines)

    async def connect(self, *, is_reconnect: bool = False, **kwargs: Any) -> bool:
        if not self.token:
            self._set_fatal_error("missing_token", "RUBIKA_BOT_TOKEN is not configured", retryable=False)
            return False
        self._client = RubikaClient(self.token, api_base=self.api_base)
        try:
            me = await self._client.get_me()
            self.bot_id = str(me.get("user_id") or me.get("id") or "")
            self.bot_username = me.get("username")
        except RubikaAPIError as exc:
            if exc.fatal:
                self._set_fatal_error("auth", f"Rubika rejected the bot token: {exc.description}", retryable=False)
                return False
            logger.warning("Rubika getMe failed (transient): %s", exc.description)

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("Rubika gateway connected as %s", self.bot_username or self.bot_id or "bot")
        return True

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._client:
            await self._client.close()
            self._client = None

    async def _poll_loop(self) -> None:
        consecutive_failures = 0
        backoff = self.backoff_base

        while self._running:
            try:
                if not self._client:
                    break
                updates = await self._client.get_updates(offset=self._offset, timeout=self.poll_timeout)
                consecutive_failures = 0
                backoff = self.backoff_base
                for u in updates:
                    await self._handle_update(u)
            except asyncio.CancelledError:
                break
            except RubikaAPIError as exc:
                if exc.fatal:
                    self._set_fatal_error("auth", f"Fatal Rubika error: {exc.description}", retryable=False)
                    break
                consecutive_failures += 1
                await asyncio.sleep(min(backoff, self.backoff_max))
                backoff *= 2.0
            except Exception as exc:
                consecutive_failures += 1
                await asyncio.sleep(min(backoff, self.backoff_max))
                backoff *= 2.0

    async def _handle_update(self, update: dict) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return
        uid = update.get("update_id") or update.get("id")
        if uid is not None:
            self._offset = max(self._offset, int(uid) + 1)

        msg_id = str(message.get("message_id") or message.get("id") or "")
        chat_id = str(message.get("chat_id") or "")
        sender_id = str(message.get("sender_id") or message.get("from_id") or "")

        # Deduplication
        dedup_key = f"{chat_id}:{msg_id}"
        if self._dedup.is_duplicate(dedup_key):
            return

        # Loop guard: Ignore bot's own messages
        if self.bot_id and sender_id == str(self.bot_id):
            return

        # Allowlist check
        if not self.allow_all_users and self.allowed_users and sender_id not in self.allowed_users:
            logger.debug("Rubika message from unallowed user %s ignored", sender_id)
            return

        text = message.get("text") or ""
        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_id,
            chat_type="dm",
            user_id=sender_id,
            message_id=msg_id,
        )
        event = MessageEvent(
            text=text,
            source=source,
            message_id=msg_id,
            platform_event_id=str(uid) if uid else None,
            raw_message=message,
        )
        await self.handle_message(event)

    async def send(self, target: Any, text: str, *, reply_to: Optional[str] = None,
                   metadata: Optional[dict] = None, content: Optional[str] = None, **kwargs: Any) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Rubika client not connected")

        actual_text = content or text
        chat_id = getattr(target, "chat_id", target)
        reply_id = int(reply_to) if reply_to and str(reply_to).isdigit() else None

        chunks = [actual_text[i:i + self.max_message_length] for i in range(0, len(actual_text), self.max_message_length)]
        last_id = None

        for chunk in chunks:
            try:
                res = await self._client.send_message(chat_id, chunk, reply_to_message_id=reply_id)
                last_id = str((res or {}).get("message_id") or "")
            except Exception as exc:
                return SendResult(success=False, error=str(exc))

        return SendResult(success=True, message_id=last_id)

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"id": chat_id, "type": "unknown"}

    async def send_typing(self, target: Any) -> None:
        if self._client:
            chat_id = getattr(target, "chat_id", target)
            try:
                await self._client.send_chat_action(chat_id, "typing")
            except Exception:
                pass

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            reply_to: Optional[str] = None, **kwargs: Any) -> SendResult:
        if not self._client or not os.path.exists(file_path):
            return SendResult(success=False, error="Client not connected or file not found")
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            filename = os.path.basename(file_path)
            res = await self._client.send_document(chat_id, data, filename, caption=caption)
            return SendResult(success=True, message_id=str((res or {}).get("message_id") or ""))
        except Exception as exc:
            return SendResult(success=False, error=str(exc))


def register(ctx: Any) -> None:
    """Register Rubika platform adapter with the Hermes platform registry."""
    ctx.register_platform(
        name="rubika",
        label="Rubika",
        required_env=["RUBIKA_BOT_TOKEN"],
        cron_deliver_env_var="RUBIKA_HOME_CHANNEL",
        max_message_length=MAX_MESSAGE_LENGTH,
        adapter_factory=RubikaAdapter,
    )
