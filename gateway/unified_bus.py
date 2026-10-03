"""Unified Message Bus and Session Resolver for Hermes Multi-Platform Architecture.

Implements the formal P0 pipeline:
Platform Adapter -> Normalizer -> Unified Message -> Message Bus -> Session Resolver -> Agent.
Includes per-session mutual exclusion (asyncio.Lock per session key) to guarantee that
concurrent messages for the same conversation thread do not race, unless explicitly decoupled.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Set

from gateway.platforms.event import MessageEvent, UnifiedMessage

logger = logging.getLogger(__name__)


@dataclass
class SessionIdentity:
    """Canonical session identity separating platform transport from agent state."""

    platform_identity: str
    conversation_id: str
    raw_user_id: Optional[str] = None
    global_user_id: str = field(init=False)
    session_id: Optional[str] = None
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        user_part = self.raw_user_id or "anon"
        self.global_user_id = f"{self.platform_identity}:{user_part}"

    @property
    def canonical_key(self) -> str:
        """Stable key for session lookup and locking."""
        return f"{self.platform_identity}:{self.conversation_id}"


class SessionResolver:
    """Resolves platform events into canonical SessionIdentity and manages per-session mutual exclusion."""

    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock_guard = asyncio.Lock()
        # Explicit identity links: maps secondary global_user_id -> primary global_user_id
        # NOTE: Identities are NEVER merged automatically by username or display name!
        self._identity_links: Dict[str, str] = {}
        # Authorized linking tokens: maps token -> authorized primary global_user_id
        self._pending_linking_grants: Dict[str, str] = {}

    def resolve_identity(self, event: MessageEvent) -> SessionIdentity:
        """Resolve a MessageEvent / UnifiedMessage to its canonical SessionIdentity."""
        platform = str(getattr(event, "platform", None) or "unknown").lower()
        chat_id = str(getattr(event, "chat_id", None) or getattr(getattr(event, "source", None), "chat_id", "") or "default")
        user_id = str(getattr(event, "user_id", None) or getattr(getattr(event, "source", None), "user_id", "") or "") or None

        identity = SessionIdentity(
            platform_identity=platform,
            conversation_id=chat_id,
            raw_user_id=user_id,
        )

        # Check explicit authorized identity link
        if identity.global_user_id in self._identity_links:
            identity.global_user_id = self._identity_links[identity.global_user_id]

        return identity

    def authorize_identity_link(self, primary_global_id: str, secondary_global_id: str) -> None:
        """Link a secondary platform identity to a primary identity with explicit authorization."""
        if not primary_global_id or not secondary_global_id:
            raise ValueError("Both primary and secondary global IDs must be non-empty")
        if primary_global_id == secondary_global_id:
            return
        self._identity_links[secondary_global_id] = primary_global_id
        logger.info("Explicitly linked identity %s -> %s", secondary_global_id, primary_global_id)

    def revoke_identity_link(self, secondary_global_id: str) -> bool:
        """Revoke a previously linked secondary identity."""
        return self._identity_links.pop(secondary_global_id, None) is not None

    def get_session_key(self, identity: SessionIdentity) -> str:
        """Derive the locking key for this identity."""
        return identity.canonical_key

    async def get_session_lock(self, session_key: str) -> asyncio.Lock:
        """Retrieve or create the mutual exclusion lock for a session key."""
        async with self._lock_guard:
            if session_key not in self._locks:
                self._locks[session_key] = asyncio.Lock()
            return self._locks[session_key]

    @contextlib.asynccontextmanager
    async def session_turn_lease(self, session_key: str, *, timeout: Optional[float] = None) -> AsyncIterator[str]:
        """Async context manager providing mutual exclusion for an agent turn.

        Ensures message A and message B for the same session do not run simultaneously.
        """
        lock = await self.get_session_lock(session_key)
        if timeout is not None:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
        else:
            await lock.acquire()
        try:
            yield session_key
        finally:
            lock.release()


MessageHandler = Callable[[UnifiedMessage], Awaitable[None]]


class UnifiedMessageBus:
    """Central unified message bus connecting platform adapters to agent consumers."""

    def __init__(self, session_resolver: Optional[SessionResolver] = None) -> None:
        self.session_resolver = session_resolver or SessionResolver()
        self._subscribers: List[tuple[Optional[str], MessageHandler]] = []
        self._active_tasks: Set[asyncio.Task] = set()
        self.total_published = 0
        self.total_processed = 0
        self.total_errors = 0

    def subscribe(self, handler: MessageHandler, platform: Optional[str] = None) -> None:
        """Register a subscriber handler.

        If platform is None, receives messages from all platforms.
        """
        self._subscribers.append((platform.lower() if platform else None, handler))

    def unsubscribe(self, handler: MessageHandler, platform: Optional[str] = None) -> int:
        """Remove a subscriber handler.

        If platform is provided, removes only subscriptions for that specific platform.
        If platform is None, removes all subscriptions for this handler.
        Returns the number of removed subscriptions.
        """
        target_p = platform.lower() if platform else None
        prev_count = len(self._subscribers)
        if target_p is not None:
            self._subscribers = [
                sub for sub in self._subscribers
                if not (sub[1] == handler and sub[0] == target_p)
            ]
        else:
            self._subscribers = [sub for sub in self._subscribers if sub[1] != handler]
        return prev_count - len(self._subscribers)

    async def publish(self, message: UnifiedMessage) -> None:
        """Publish an incoming unified message through the bus to matching subscribers."""
        self.total_published += 1
        platform = str(getattr(message, "platform", None) or "").lower()

        # Stamp correlation and trace IDs if missing
        if not getattr(message, "request_id", None):
            message.request_id = uuid.uuid4().hex
        if not getattr(message, "trace_id", None):
            message.trace_id = uuid.uuid4().hex
        if not getattr(message, "correlation_id", None):
            message.correlation_id = uuid.uuid4().hex

        # Dispatch to matching subscribers
        matching_handlers = [
            h for p, h in self._subscribers if p is None or p == platform
        ]

        if not matching_handlers:
            logger.debug("UnifiedMessageBus: No subscribers for platform=%s", platform)
            return

        for handler in matching_handlers:
            try:
                await handler(message)
                self.total_processed += 1
            except Exception as exc:
                self.total_errors += 1
                logger.error("UnifiedMessageBus handler error for platform=%s: %s", platform, exc, exc_info=exc)


# Global singleton instance for easy cross-module sharing
global_session_resolver = SessionResolver()
global_message_bus = UnifiedMessageBus(global_session_resolver)
