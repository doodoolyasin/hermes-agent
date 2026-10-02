import asyncio
import pytest
from gateway.unified_bus import UnifiedMessageBus, SessionResolver, SessionIdentity
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, Platform


@pytest.mark.asyncio
async def test_cross_platform_identity_isolation():
    resolver = SessionResolver()

    # User 101 on Bale
    ev_bale = MessageEvent(
        text="Hi from Bale",
        source=SessionSource(platform=Platform.LOCAL, chat_id="chat_101", user_id="user_101")
    )
    ev_bale.platform = "bale"

    # User 101 on Rubika
    ev_rubika = MessageEvent(
        text="Hi from Rubika",
        source=SessionSource(platform=Platform.LOCAL, chat_id="chat_101", user_id="user_101")
    )
    ev_rubika.platform = "rubika"

    id_bale = resolver.resolve_identity(ev_bale)
    id_rubika = resolver.resolve_identity(ev_rubika)

    # Global IDs and Canonical keys MUST be completely isolated
    assert id_bale.global_user_id == "bale:user_101"
    assert id_rubika.global_user_id == "rubika:user_101"
    assert id_bale.canonical_key == "bale:chat_101"
    assert id_rubika.canonical_key == "rubika:chat_101"
    assert id_bale.canonical_key != id_rubika.canonical_key


@pytest.mark.asyncio
async def test_session_turn_lease_mutual_exclusion():
    resolver = SessionResolver()
    session_key = "bale:chat_999"

    active_count = 0
    max_concurrent_seen = 0

    async def worker(task_idx: int):
        nonlocal active_count, max_concurrent_seen
        async with resolver.session_turn_lease(session_key):
            active_count += 1
            if active_count > max_concurrent_seen:
                max_concurrent_seen = active_count
            await asyncio.sleep(0.02)
            active_count -= 1

    # Launch 10 concurrent requests for the exact same session
    await asyncio.gather(*(worker(i) for i in range(10)))

    # In a given session, concurrent count must NEVER exceed 1!
    assert max_concurrent_seen == 1
    assert active_count == 0


@pytest.mark.asyncio
async def test_session_turn_lease_exception_no_lock_poisoning():
    resolver = SessionResolver()
    session_key = "bale:chat_error"

    # First turn raises exception
    with pytest.raises(ValueError):
        async with resolver.session_turn_lease(session_key):
            raise ValueError("Turn failed mid-execution")

    # Second turn must immediately acquire the lock without deadlocking
    acquired = False
    async with resolver.session_turn_lease(session_key, timeout=1.0):
        acquired = True

    assert acquired is True
