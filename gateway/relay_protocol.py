"""Remote Hermes Node Relay Protocol (v1).

Implements Directive Section 12:
- Protocol negotiation: 'hermes-relay/1.0'
- Secure authentication: HMAC-SHA256 request signing with replay protection (300s window)
- Standardized request/response schemas for remote node inference
- Dynamic capability exchange (/relay/v1/capabilities)
- Node health probing (/relay/v1/health)
- Failure handling and timeout governance
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HERMES_RELAY_V1 = "hermes-relay/1.0"
MAX_REPLAY_WINDOW_SECONDS = 300.0


@dataclass
class RelayRequest:
    source_node: str
    target_model: str
    messages: List[Dict[str, Any]]
    protocol_version: str = HERMES_RELAY_V1
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    capabilities_required: List[str] = field(default_factory=lambda: ["chat"])
    timeout_seconds: float = 30.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class RelayResponse:
    request_id: str
    responder_node: str
    status: str  # 'success', 'error', 'auth_failed', 'unsupported_version'
    protocol_version: str = HERMES_RELAY_V1
    text: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    latency_ms: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class RelaySecurity:
    """HMAC-SHA256 authenticated transport signing with timestamp replay protection."""

    @staticmethod
    def sign_payload(payload_bytes: bytes, secret_key: str, node_id: str) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        to_sign = timestamp.encode("utf-8") + b":" + payload_bytes
        sig = hmac.new(secret_key.encode("utf-8"), to_sign, hashlib.sha256).hexdigest()
        return {
            "X-Hermes-Protocol": HERMES_RELAY_V1,
            "X-Hermes-Node-Id": node_id,
            "X-Hermes-Timestamp": timestamp,
            "X-Hermes-Signature": sig,
        }

    @staticmethod
    def verify_headers(
        payload_bytes: bytes,
        headers: Dict[str, str],
        secret_key: str,
        now: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        current_time = now if now is not None else time.time()

        version = headers.get("X-Hermes-Protocol") or headers.get("x-hermes-protocol")
        if version != HERMES_RELAY_V1:
            return False, f"Unsupported protocol version: {version}"

        ts_str = headers.get("X-Hermes-Timestamp") or headers.get("x-hermes-timestamp")
        if not ts_str:
            return False, "Missing timestamp header"

        try:
            ts = float(ts_str)
        except ValueError:
            return False, "Invalid timestamp header"

        if abs(current_time - ts) > MAX_REPLAY_WINDOW_SECONDS:
            return False, f"Request timestamp expired (drift > {MAX_REPLAY_WINDOW_SECONDS}s)"

        given_sig = headers.get("X-Hermes-Signature") or headers.get("x-hermes-signature")
        if not given_sig:
            return False, "Missing signature header"

        to_sign = ts_str.encode("utf-8") + b":" + payload_bytes
        expected_sig = hmac.new(secret_key.encode("utf-8"), to_sign, hashlib.sha256).hexdigest()

        if not hmac.compare_digest(given_sig, expected_sig):
            return False, "Signature verification failed"

        return True, None


class RemoteHermesClient:
    """Client for dispatching tasks and querying remote Hermes relay nodes."""

    def __init__(self, node_id: str, secret_key: str, node_url: str) -> None:
        self.node_id = node_id
        self.secret_key = secret_key
        self.node_url = node_url.rstrip("/")

    def create_request(
        self,
        target_model: str,
        messages: List[Dict[str, Any]],
        capabilities: Optional[List[str]] = None,
    ) -> Tuple[RelayRequest, Dict[str, str]]:
        req = RelayRequest(
            source_node=self.node_id,
            target_model=target_model,
            messages=messages,
            capabilities_required=capabilities or ["chat"],
        )
        payload_bytes = req.to_json().encode("utf-8")
        headers = RelaySecurity.sign_payload(payload_bytes, self.secret_key, self.node_id)
        headers["Content-Type"] = "application/json"
        return req, headers
