import time
import pytest
from gateway.relay_protocol import (
    RelaySecurity,
    RelayRequest,
    RelayResponse,
    HERMES_RELAY_V1,
    RemoteHermesClient,
)


def test_relay_signing_and_verification():
    secret = "super_secret_shared_key_123"
    client = RemoteHermesClient("node-alpha", secret, "https://relay.hermes.ir")

    req, headers = client.create_request("llama3.2", [{"role": "user", "content": "ping"}])
    payload_bytes = req.to_json().encode("utf-8")

    # Valid verification
    ok, err = RelaySecurity.verify_headers(payload_bytes, headers, secret)
    assert ok is True
    assert err is None


def test_relay_rejects_tampered_payload():
    secret = "super_secret_shared_key_123"
    client = RemoteHermesClient("node-alpha", secret, "https://relay.hermes.ir")

    req, headers = client.create_request("llama3.2", [{"role": "user", "content": "ping"}])
    tampered_bytes = b'{"tampered": true}'

    ok, err = RelaySecurity.verify_headers(tampered_bytes, headers, secret)
    assert ok is False
    assert "Signature verification failed" in err


def test_relay_replay_protection():
    secret = "super_secret_shared_key_123"
    client = RemoteHermesClient("node-alpha", secret, "https://relay.hermes.ir")

    req, headers = client.create_request("llama3.2", [{"role": "user", "content": "ping"}])
    payload_bytes = req.to_json().encode("utf-8")

    # Simulate request from 400 seconds ago (outside 300s window)
    now = time.time()
    old_time = now + 400.0

    ok, err = RelaySecurity.verify_headers(payload_bytes, headers, secret, now=old_time)
    assert ok is False
    assert "timestamp expired" in err
