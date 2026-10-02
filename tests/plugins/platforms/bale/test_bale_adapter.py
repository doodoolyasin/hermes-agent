"""Tests for the Bale gateway plugin adapters.

All network access is mocked at the single HTTP seam (``BaleClient._request``'s
transport), so these tests never touch the internet.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.bale import adapter as bale


# ── helpers ──────────────────────────────────────────────────────────
def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("BALE_BOT_TOKEN", "BALE_API_BASE", "BALE_POLL_TIMEOUT",
                 "BALE_MAX_FAILURES", "BALE_BACKOFF_BASE", "BALE_BACKOFF_MAX",
                 "BALE_PARSE_MODE", "BALE_MARKDOWN", "BALE_HOME_CHANNEL"):
        monkeypatch.delenv(name, raising=False)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, raw=False):
        self.status_code = status_code
        self._payload = payload
        self._raw = raw

    def json(self):
        if self._raw:
            raise ValueError("not json")
        return self._payload


class FakeHTTP:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def post(self, url, json=None, data=None, files=None, timeout=None):
        self.calls.append({"url": url, "json": json, "data": data, "files": files})
        if not self._responses:
            raise AssertionError("unexpected extra HTTP call")
        return self._responses.pop(0)

    async def aclose(self):
        return None


def make_client(responses):
    client = bale.BaleClient("123456:secret-token", api_base="https://tapi.bale.ai")
    client._client = FakeHTTP(responses)
    return client


def make_adapter(extra=None, **cfg):
    cfg.setdefault("enabled", True)
    return bale.BaleAdapter(PlatformConfig(extra=dict(extra or {}), **cfg))


# ── registration / static surface ────────────────────────────────────
def test_register_platform_shape():
    captured = {}

    class Ctx:
        def register_platform(self, **kwargs):
            captured.update(kwargs)

    bale.register(Ctx())
    assert captured["name"] == "bale"
    assert captured["label"] == "Bale"
    assert captured["required_env"] == ["BALE_BOT_TOKEN"]
    assert captured["allowed_users_env"] == "BALE_ALLOWED_USERS"
    assert captured["allow_all_env"] == "BALE_ALLOW_ALL_USERS"
    assert captured["cron_deliver_env_var"] == "BALE_HOME_CHANNEL"
    assert captured["max_message_length"] == 4096
    assert callable(captured["adapter_factory"])
    assert callable(captured["standalone_sender_fn"])
    assert captured["is_connected"] is bale.is_connected


def test_platform_member_resolves():
    assert Platform("bale") is Platform("bale")
    assert Platform("bale").value == "bale"


def test_redact_never_leaks_token():
    out = bale._redact("123456:AAF-verysecretvalue")
    assert "verysecretvalue" not in out
    assert "secret" not in out
    assert bale._redact(None) == ""
    assert bale._redact("short") == "[REDACTED]"


# ── error classification ─────────────────────────────────────────────
def test_classify_rate_limit_uses_retry_after():
    err = bale._classify(429, 429, {"description": "Too Many Requests",
                                    "parameters": {"retry_after": 7}})
    assert err.retryable is True and err.fatal is False
    assert err.retry_after == 7.0


def test_classify_auth_is_fatal():
    err = bale._classify(401, 401, {"description": "Unauthorized"})
    assert err.fatal is True and err.retryable is False


def test_classify_server_error_retryable():
    err = bale._classify(503, None, {})
    assert err.retryable is True and err.fatal is False


def test_classify_bad_request_not_retryable():
    err = bale._classify(400, 400, {"description": "bad request"})
    assert err.retryable is False and err.fatal is False


# ── BaleClient request mapping ───────────────────────────────────────
def test_request_returns_result():
    client = make_client([FakeResponse(200, {"ok": True, "result": {"id": 42,
                                                                    "username": "bot"}})])
    assert run(client.get_me())["id"] == 42


def test_request_ok_false_raises():
    client = make_client([FakeResponse(400, {"ok": False, "error_code": 400,
                                             "description": "Bad Request"})])
    with pytest.raises(bale.BaleAPIError):
        run(client.get_me())


def test_request_non_json_raises():
    client = make_client([FakeResponse(502, raw=True)])
    with pytest.raises(bale.BaleAPIError):
        run(client.get_me())


def test_request_transport_error_is_retryable():
    class Boom:
        async def post(self, *a, **k):
            raise OSError("connection refused")

    client = bale.BaleClient("1:a")
    client._client = Boom()
    with pytest.raises(bale.BaleAPIError) as excinfo:
        run(client.get_me())
    assert excinfo.value.retryable is True


def test_get_updates_uses_offset_and_timeout():
    client = make_client([FakeResponse(200, {"ok": True, "result": []})])
    run(client.get_updates(17, timeout=5))
    sent = client._client.calls[0]
    assert sent["json"]["offset"] == 17
    assert sent["json"]["timeout"] == 5
    assert sent["url"].endswith("/getUpdates")


# ── adapter configuration ────────────────────────────────────────────
def test_adapter_reads_token_from_extra():
    assert make_adapter({"token": "tok-from-extra"}).token == "tok-from-extra"


def test_adapter_reads_token_and_knobs_from_env(monkeypatch):
    monkeypatch.setenv("BALE_BOT_TOKEN", "env-token")
    monkeypatch.setenv("BALE_MAX_FAILURES", "3")
    monkeypatch.setenv("BALE_BACKOFF_MAX", "12.5")
    adapter = make_adapter()
    assert adapter.token == "env-token"
    assert adapter.max_attempts == 3
    assert adapter.backoff_max == 12.5


def test_adapter_default_knobs_match_manifest():
    adapter = make_adapter({"token": "t"})
    assert adapter.max_attempts == 8
    assert adapter.backoff_max == 60.0


def test_markdown_toggle(monkeypatch):
    default = make_adapter({"token": "t"})
    assert "**" not in default.format_message("**bold**")
    monkeypatch.setenv("BALE_MARKDOWN", "1")
    passthrough = make_adapter({"token": "t"})
    assert passthrough.format_message("**bold**") == "**bold**"


def test_chunk_splits_at_limit():
    adapter = make_adapter({"token": "t"})
    adapter.max_message_length = 10
    chunks = adapter._chunk("x" * 25)
    assert len(chunks) == 3 and all(len(c) <= 10 for c in chunks)


# ── inbound handling ─────────────────────────────────────────────────
def _update(update_id=1, chat_id=555, chat_type="private", text="hello",
            from_id=999, message_id=11):
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "from": {"id": from_id, "first_name": "Ada", "username": "ada"},
            "date": 1700000000,
            "chat": {"id": chat_id, "type": chat_type, "title": "Group X"},
            "text": text,
        },
    }


def test_handle_update_builds_source_and_emits_event():
    adapter = make_adapter({"token": "t"})
    captured = {}
    adapter.build_source = lambda **kw: (captured.update(kw), SimpleNamespace(**kw))[1]
    events = []

    async def fake_handle(event):
        events.append(event)

    adapter.handle_message = fake_handle
    run(adapter._handle_update(_update(chat_type="private")))
    assert captured["chat_id"] == "555"
    assert captured["chat_type"] == "dm"
    assert captured["user_id"] == "999"
    assert events and events[0].text == "hello"
    assert events[0].message_type == bale.MessageType.TEXT


def test_handle_update_group_chat_type():
    adapter = make_adapter({"token": "t"})
    captured = {}
    adapter.build_source = lambda **kw: (captured.update(kw), SimpleNamespace(**kw))[1]

    async def fake_handle(event):
        return None

    adapter.handle_message = fake_handle
    run(adapter._handle_update(_update(chat_type="group")))
    assert captured["chat_type"] == "group"


def test_handle_update_dedups_repeated_update_id():
    adapter = make_adapter({"token": "t"})
    adapter.build_source = lambda **kw: SimpleNamespace(**kw)
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle
    upd = _update(update_id=7)
    run(adapter._handle_update(upd))
    run(adapter._handle_update(upd))
    assert len(seen) == 1


def test_handle_update_ignores_self_messages():
    adapter = make_adapter({"token": "t"})
    adapter.bot_id = 999  # same as sender id
    adapter.build_source = lambda **kw: SimpleNamespace(**kw)
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle
    run(adapter._handle_update(_update(from_id=999)))
    assert seen == []


# ── outbound ─────────────────────────────────────────────────────────
def test_send_success():
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([FakeResponse(200, {"ok": True, "result": {"message_id": 5}})])
    result = run(adapter.send("555", "hi"))
    assert result.success is True
    assert result.message_id == "5"
    assert adapter._client._client.calls[0]["json"]["chat_id"] == "555"


def test_send_fatal_error_returns_failure_without_retry():
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([FakeResponse(401, {"ok": False, "error_code": 401,
                                                      "description": "Unauthorized"})])
    result = run(adapter.send("555", "hi"))
    assert result.success is False
    assert "401" in result.error or "Unauthorized" in result.error
    assert len(adapter._client._client.calls) == 1


def test_send_retries_on_server_error_then_succeeds(monkeypatch):
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([
        FakeResponse(500, {"ok": False, "error_code": 500, "description": "boom"}),
        FakeResponse(200, {"ok": True, "result": {"message_id": 9}}),
    ])

    async def no_sleep(_):
        return None

    monkeypatch.setattr(bale.asyncio, "sleep", no_sleep)
    result = run(adapter.send("555", "hi"))
    assert result.success is True and result.message_id == "9"
    assert len(adapter._client._client.calls) == 2


def test_send_respects_rate_limit_retry_after(monkeypatch):
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([
        FakeResponse(429, {"ok": False, "error_code": 429, "description": "slow down",
                           "parameters": {"retry_after": 3}}),
        FakeResponse(200, {"ok": True, "result": {"message_id": 1}}),
    ])
    slept = []

    async def rec_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(bale.asyncio, "sleep", rec_sleep)
    result = run(adapter.send("555", "hi"))
    assert result.success is True
    assert slept and slept[0] == 3.0


def test_send_gives_up_after_cap(monkeypatch):
    adapter = make_adapter({"token": "t"})
    adapter.max_attempts = 2
    adapter._client = make_client([
        FakeResponse(500, {"ok": False, "error_code": 500, "description": "boom"}),
        FakeResponse(500, {"ok": False, "error_code": 500, "description": "boom"}),
    ])

    async def no_sleep(_):
        return None

    monkeypatch.setattr(bale.asyncio, "sleep", no_sleep)
    result = run(adapter.send("555", "hi"))
    assert result.success is False
    assert len(adapter._client._client.calls) == 2


def test_send_not_connected():
    adapter = make_adapter({"token": "t"})
    result = run(adapter.send("555", "hi"))
    assert result.success is False and "not connected" in result.error


def test_send_document_uses_sendDocument(tmp_path):
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([FakeResponse(200, {"ok": True, "result": {"message_id": 3}})])
    f = tmp_path / "report.txt"
    f.write_text("data")
    result = run(adapter.send_document("555", str(f), caption="here"))
    assert result.success is True
    call = adapter._client._client.calls[0]
    assert call["url"].endswith("/sendDocument")
    assert "document" in call["files"]


def test_send_image_uses_sendPhoto():
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([FakeResponse(200, {"ok": True, "result": {"message_id": 4}})])
    result = run(adapter.send_image("555", "https://example.com/a.png"))
    assert result.success is True
    call = adapter._client._client.calls[0]
    assert call["url"].endswith("/sendPhoto")
    assert call["json"]["photo"] == "https://example.com/a.png"


# ── standalone / requirements ────────────────────────────────────────
def test_standalone_send_without_token_is_error():
    res = run(bale._standalone_send("555", "hi"))
    assert isinstance(res, dict)
    assert res.get("ok") is False or "error" in res


def test_check_requirements_reports_missing_token():
    ok, detail = bale.check_requirements()
    assert ok is False and "BALE_BOT_TOKEN" in detail


def test_env_enablement_none_without_token():
    assert bale._env_enablement() is None


def test_env_enablement_with_token(monkeypatch):
    monkeypatch.setenv("BALE_BOT_TOKEN", "from-env")
    extra = bale._env_enablement()
    assert extra and extra["token"] == "from-env"


def test_apply_yaml_config_bridges_keys(monkeypatch):
    cfg = {
        "token": "tok123",
        "home_channel": "12345",
        "allowed_users": ["111", "222"],
        "markdown": True,
    }
    extra = bale._apply_yaml_config({}, cfg)
    assert extra["token"] == "tok123"
    assert extra["home_channel"] == "12345"
    assert extra["markdown"] is True


def test_standalone_send_with_pconfig(monkeypatch):
    class FakeConfig:
        token = "test-token"
        extra = {"api_base": "https://fake.bale.ai"}

    class FakeClient:
        def __init__(self, token, api_base=None):
            self.token = token
            self.api_base = api_base

        async def send_message(self, chat_id, text):
            return {"message_id": 999, "chat_id": chat_id, "text": text}

        async def close(self):
            pass

    monkeypatch.setattr(bale, "BaleClient", FakeClient)
    res = run(bale._standalone_send(FakeConfig(), "777", "hello bale", thread_id="10"))
    assert res.get("ok") is True
    assert res.get("result", {}).get("message_id") == 999


def test_edit_message():
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([
        FakeResponse(200, {"ok": True, "result": {"message_id": 42}}),
    ])
    result = run(adapter.edit_message("100", "42", "updated content"))
    assert result.success is True
    assert result.message_id == "42"
    call = adapter._client._client.calls[0]
    assert call["url"].endswith("/editMessageText")
    assert call["json"]["text"] == "updated content"


def test_delete_message():
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([
        FakeResponse(200, {"ok": True, "result": True}),
    ])
    success = run(adapter.delete_message("100", "42"))
    assert success is True
    call = adapter._client._client.calls[0]
    assert call["url"].endswith("/deleteMessage")
    assert call["json"]["message_id"] == 42


def test_send_chat_action_as_form():
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([
        FakeResponse(200, {"ok": True, "result": True}),
    ])
    run(adapter.send_typing("100"))
    call = adapter._client._client.calls[0]
    assert call["url"].endswith("/sendChatAction")
    assert call["data"]["action"] == "typing"
    assert call["data"]["chat_id"] == "100"


def test_send_with_content_kwarg():
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([
        FakeResponse(200, {"ok": True, "result": {"message_id": 77}}),
    ])
    result = run(adapter.send(chat_id="100", content="hello from content"))
    assert result.success is True
    assert result.message_id == "77"
    call = adapter._client._client.calls[0]
    assert call["json"]["text"] == "hello from content"


def test_bale_send_dashboard():
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([
        FakeResponse(200, {"ok": True, "result": {"message_id": 101}}),
    ])
    result = run(adapter.send_dashboard("100"))
    assert result.success is True
    assert result.message_id == "101"
    call = adapter._client._client.calls[0]
    assert "reply_markup" in call["json"]
    assert "inline_keyboard" in call["json"]["reply_markup"]


def test_bale_send_model_picker_and_callback():
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([
        # 1. send_model_picker
        FakeResponse(200, {"ok": True, "result": {"message_id": 202}}),
        # 2. answerCallbackQuery
        FakeResponse(200, {"ok": True, "result": True}),
        # 3. editMessageText
        FakeResponse(200, {"ok": True, "result": {"message_id": 202}}),
    ])

    switched = []
    async def fake_on_model_selected(chat_id, model_id, provider_slug):
        switched.append((chat_id, model_id, provider_slug))
        return f"Switched to {model_id}"

    providers = [{
        "name": "Free AI",
        "slug": "custom",
        "models": ["openai-fast", "gpt-oss-20b"],
        "is_current": True,
    }]

    result = run(adapter.send_model_picker(
        "100", providers, "openai-fast", "custom", "test_key", fake_on_model_selected
    ))
    assert result.success is True

    # Simulate callback query when user clicks the 2nd model (gpt-oss-20b -> mm:1)
    cb_update = {
        "update_id": 999,
        "callback_query": {
            "id": "query_123",
            "from": {"id": 100},
            "message": {
                "message_id": 202,
                "chat": {"id": 100},
            },
            "data": "mm:1",
        }
    }
    run(adapter._handle_update(cb_update))
    assert len(switched) == 1
    assert switched[0] == ("100", "gpt-oss-20b", "custom")


def test_bale_choice_picker():
    adapter = make_adapter({"token": "t"})
    adapter._client = make_client([
        FakeResponse(200, {"ok": True, "result": {"message_id": 303}}),
        FakeResponse(200, {"ok": True, "result": True}),
        FakeResponse(200, {"ok": True, "result": {"message_id": 303}}),
    ])

    chosen = []
    async def fake_on_choice(chat_id, val):
        chosen.append((chat_id, val))
        return f"Picked {val}"

    choices = [{"label": "High", "value": "high"}, {"label": "Low", "value": "low"}]
    res = run(adapter.send_choice_picker("100", "Pick level", choices, "sess", fake_on_choice))
    assert res.success is True

    # Callback
    cb = {
        "update_id": 1001,
        "callback_query": {
            "id": "q456",
            "from": {"id": 100},
            "message": {"message_id": 303, "chat": {"id": 100}},
            "data": "cp:0",
        }
    }
    run(adapter._handle_update(cb))
    assert chosen == [("100", "high")]



