import asyncio
import pytest
from gateway.unified_bus import UnifiedMessageBus, SessionResolver, SessionIdentity
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_bus_session_timeout_and_lock_recovery():
    resolver = SessionResolver()
    session_key = "bale:user_blocked"

    # 1. Hold lease in task 1
    acquired = asyncio.Event()
    release = asyncio.Event()

    async def long_holder():
        async with resolver.session_turn_lease(session_key):
            acquired.set()
            await release.wait()

    holder_task = asyncio.create_task(long_holder())
    await acquired.wait()

    # 2. Task 2 attempts to acquire with short timeout -> should raise TimeoutError
    with pytest.raises(asyncio.TimeoutError):
        async with resolver.session_turn_lease(session_key, timeout=0.05):
            pytest.fail("Should not acquire held lock")

    # 3. Release task 1
    release.set()
    await holder_task

    # 4. Now Task 3 must acquire cleanly without hanging
    async with resolver.session_turn_lease(session_key, timeout=0.5):
        assert True


@pytest.mark.asyncio
async def test_bus_independent_session_concurrency_no_starvation():
    bus = UnifiedMessageBus()
    events_log = []

    async def subscriber(msg):
        # Record arrival and simulate variable processing time
        events_log.append(f"start:{msg.text}")
        if "slow" in msg.text:
            await asyncio.sleep(0.08)
        else:
            await asyncio.sleep(0.01)
        events_log.append(f"done:{msg.text}")

    bus.subscribe(subscriber)

    m1 = MessageEvent(text="slow_msg", source=SessionSource(platform="bale", chat_id="1", user_id="1"))
    m2 = MessageEvent(text="fast_msg", source=SessionSource(platform="rubika", chat_id="2", user_id="2"))

    # Dispatch both concurrently
    t1 = asyncio.create_task(bus.publish(m1))
    t2 = asyncio.create_task(bus.publish(m2))

    await asyncio.gather(t1, t2)

    assert bus.total_published == 2
    assert bus.total_processed == 2
    assert bus.total_errors == 0
    # Fast message must finish before slow message finishes!
    done_fast_idx = events_log.index("done:fast_msg")
    done_slow_idx = events_log.index("done:slow_msg")
    assert done_fast_idx < done_slow_idx
