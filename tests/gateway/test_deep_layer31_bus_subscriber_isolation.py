import asyncio
import pytest
from gateway.unified_bus import UnifiedMessageBus
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_bus_multi_subscriber_exception_isolation():
    bus = UnifiedMessageBus()
    received_by_h2 = []

    async def crashing_handler(msg):
        raise RuntimeError("Handler 1 crashed unexpectedly!")

    async def healthy_handler(msg):
        received_by_h2.append(msg.text)

    bus.subscribe(crashing_handler)
    bus.subscribe(healthy_handler)

    msg = MessageEvent(
        text="critical update",
        source=SessionSource(platform="bale", chat_id="123", user_id="123"),
    )

    # Should not raise exception
    await bus.publish(msg)

    assert bus.total_published == 1
    assert bus.total_processed == 1  # healthy handler processed
    assert bus.total_errors == 1     # crashing handler incremented error
    assert received_by_h2 == ["critical update"]


@pytest.mark.asyncio
async def test_bus_fine_grained_unsubscribe_lifecycle():
    bus = UnifiedMessageBus()
    bale_events = []
    rubika_events = []

    async def bale_sink(msg):
        bale_events.append(msg.text)

    async def common_sink(msg):
        rubika_events.append(msg.text)

    # Register
    bus.subscribe(bale_sink, platform="bale")
    bus.subscribe(common_sink, platform="rubika")
    bus.subscribe(common_sink, platform="bale")

    # 1. Dispatch bale message -> both bale_sink and common_sink receive
    m_bale = MessageEvent(
        text="bale 1",
        source=SessionSource(platform="bale", chat_id="1", user_id="1"),
    )
    await bus.publish(m_bale)
    assert bale_events == ["bale 1"]
    assert rubika_events == ["bale 1"]

    # 2. Unsubscribe common_sink from bale only
    removed = bus.unsubscribe(common_sink, platform="bale")
    assert removed == 1

    # 3. Dispatch second bale message -> only bale_sink receives
    m_bale2 = MessageEvent(
        text="bale 2",
        source=SessionSource(platform="bale", chat_id="1", user_id="1"),
    )
    await bus.publish(m_bale2)
    assert bale_events == ["bale 1", "bale 2"]
    assert rubika_events == ["bale 1"]  # did NOT receive bale 2!

    # 4. Dispatch rubika message -> common_sink still receives it!
    m_rubika = MessageEvent(
        text="rubika 1",
        source=SessionSource(platform="rubika", chat_id="2", user_id="2"),
    )
    await bus.publish(m_rubika)
    assert rubika_events == ["bale 1", "rubika 1"]

    # 5. Unsubscribe non-existent
    assert bus.unsubscribe(lambda m: None) == 0
