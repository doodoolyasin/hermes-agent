import time
import pytest
from gateway.relay_protocol import (
    RelaySecurity,
    RelayRequest,
    RelayResponse,
    HERMES_RELAY_V1,
    RemoteHermesClient,
)


def test_relay_fuzzing_future_timestamp_replay_attack():
    secret = "fuzz_secret_key_888"
    client = RemoteHermesClient("node-1", secret, "http://node2.local")

    req, headers = client.create_request("modelA", [{"role": "user", "content": "fuzz"}])
    payload_bytes = req.to_json().encode("utf-8")

    # Future timestamp attack (+10,000 seconds into the future)
    headers["X-Hermes-Timestamp"] = str(int(time.time()) + 10000)

    ok, err = RelaySecurity.verify_headers(payload_bytes, headers, secret)
    assert ok is False
    assert "timestamp expired" in err.lower() or "drift" in err.lower()


def test_relay_fuzzing_lowercase_http2_headers():
    secret = "fuzz_secret_key_888"
    client = RemoteHermesClient("node-1", secret, "http://node2.local")

    req, headers = client.create_request("modelA", [{"role": "user", "content": "test http2"}])
    payload_bytes = req.to_json().encode("utf-8")

    # Simulate HTTP/2 all-lowercase headers
    h2_headers = {k.lower(): v for k, v in headers.items()}

    ok, err = RelaySecurity.verify_headers(payload_bytes, h2_headers, secret)
    assert ok is True
    assert err is None


def test_relay_fuzzing_persian_and_multilingual_payloads():
    secret = "persian_test_key_xyz"
    client = RemoteHermesClient("node-tehran", secret, "http://relay.internal")

    complex_persian_text = (
        "سلام! این یک پیام آزمایشی با نویسه‌های ترکیبی و نیم‌فاصله‌دار است: "
        "می‌شود، آمده‌اید، 🌸✨ 123456۷۸۹۰ \u200c\u200d\u200e\u200f"
    )

    req, headers = client.create_request("persian-model", [
        {"role": "user", "content": complex_persian_text}
    ])
    payload_bytes = req.to_json().encode("utf-8")

    # Valid verification
    ok, err = RelaySecurity.verify_headers(payload_bytes, headers, secret)
    assert ok is True

    # Mutate 1 byte in the Persian UTF-8 stream
    mutated_bytes = bytearray(payload_bytes)
    mutated_bytes[20] = (mutated_bytes[20] + 1) % 256

    ok_mut, err_mut = RelaySecurity.verify_headers(bytes(mutated_bytes), headers, secret)
    assert ok_mut is False
    assert "Signature verification failed" in err_mut


def test_relay_fuzzing_empty_and_garbage_headers():
    secret = "some_secret"
    payload = b'{"empty": true}'

    # Missing headers
    ok, err = RelaySecurity.verify_headers(payload, {}, secret)
    assert ok is False

    # Garbage protocol version
    bad_proto_headers = {
        "X-Hermes-Protocol": "hermes-relay/999.0",
        "X-Hermes-Timestamp": str(int(time.time())),
        "X-Hermes-Signature": "abc",
    }
    ok, err = RelaySecurity.verify_headers(payload, bad_proto_headers, secret)
    assert ok is False
    assert "Unsupported protocol version" in err
