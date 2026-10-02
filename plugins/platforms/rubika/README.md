# Rubika Platform Adapter for Hermes

Production-grade Rubika messenger integration for Hermes Agent.

## Capabilities & Architecture
- REST/HTTPS transport with bounded long-polling (`getUpdates`).
- Automatic deduplication and offset tracking.
- Rate limiting and retry backoff.
- Capability Matrix:
  - Text messaging & replies: Supported
  - Media (Photos, Documents, Voice, Video): Supported
  - Typing indicators (`sendChatAction`): Supported
  - Inline keyboards & Callbacks: Unsupported by official Rubika bot API -> Capability-aware fallback to structured numbered text menus.
