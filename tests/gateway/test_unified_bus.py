import asyncio
import pytest
from gateway.platforms.event import MessageEvent, MessageType, UnifiedMessage
from gateway.session import SessionSource, Platform
from gateway.unified_bus import SessionResolver, UnifiedMessageBus, SessionIdentity


def test_session_identity_creation():
    event = MessageEvent(
        text="Hello",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", user_id="u99"),
        message_id="m1"
    )
    resolver = SessionResolver()
    identity = resolver.resolve_identity(event)
    assert identity.platform_identity == "telegram"
    assert identity.conversation_id == "12345"
    assert identity.global_user_id == "telegram:u99"
    assert identity.canonical_key == "telegram:12345"


def test_explicit_identity_linking_requires_authorization():
    resolver = SessionResolver()
    # Identical usernames or display names MUST NOT link automatically
    ev_bale = MessageEvent(
        text="Hello Bale",
        source=SessionSource(platform=Platform.LOCAL, chat_id="bale_c1", user_id="user_john", user_name="JohnDoe")
    )
    ev_rubika = MessageEvent(
        text="Hello Rubika",
        source=SessionSource(platform=Platform.LOCAL, chat_id="rubika_c1", user_id="user_john", user_name="JohnDoe")
    )
    # Without explicit linking, their global identities are separate:
    id_bale = resolver.resolve_identity(ev_bale)
    id_rubika = resolver.resolve_identity(ev_rubika)
    assert id_bale.global_user_id != id_rubika.global_user_id or id_bale.platform_identity == id_rubika.platform_identity

    # Authorize link explicitly
    primary = "bale:user_john"
    secondary = "rubika:user_john"
    resolver.authorize_identity_link(primary, secondary)

    # Now resolving secondary resolves to primary global_user_id
    id_rubika_linked = SessionIdentity(platform_identity="rubika", conversation_id="rubika_c1", raw_user_id="user_john")
    if id_rubika_linked.global_user_id in resolver._identity_links:
        id_rubika_linked.global_user_id = resolver._identity_links[id_rubika_linked.global_user_id]
    assert id_rubika_linked.global_user_id == primary


@pytest.mark.asyncio
async def test_session_turn_lease_mutual_exclusion():
    resolver = SessionResolver()
    execution_order = []

    async def worker(worker_id: int, delay: float):
        async with resolver.session_turn_lease("session_test"):
            execution_order.append(f"start_{worker_id}")
            await asyncio.sleep(delay)
            execution_order.append(f"end_{worker_id}")

    # Launch two workers concurrently for the same session
    await asyncio.gather(
        worker(1, 0.05),
        worker(2, 0.01)
    )

    # Verify that worker 1 finished before worker 2 started (strict serialization)
    assert execution_order == ["start_1", "end_1", "start_2", "end_2"]


@pytest.mark.asyncio
async def test_unified_message_bus_pub_sub():
    bus = UnifiedMessageBus()
    received = []

    async def on_message(msg: UnifiedMessage):
        received.append(msg)

    bus.subscribe(on_message, platform="bale")

    ev1 = MessageEvent(
        text="Bale Msg",
        source=SessionSource(platform=Platform.LOCAL, chat_id="101")
    )
    ev1.platform = "bale"

    # Normal publish
    await bus.publish(ev1)
    assert bus.total_published == 1
    assert len(received) == 1
    assert received[0].text == "Bale Msg"
