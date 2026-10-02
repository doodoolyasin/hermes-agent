"""Platform capability matrix and descriptor for Hermes platform adapters.

Every platform adapter declares its dynamic or static capabilities via PlatformCapabilities,
allowing the gateway, agent, and user-facing UI to make capability-aware decisions (such as
graceful fallback from inline buttons to text menus when callbacks/keyboards are unsupported).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class PlatformCapabilities:
    """Declared capabilities of a messaging platform adapter."""

    markdown: bool = False
    message_edit: bool = False
    message_delete: bool = False
    inline_keyboard: bool = False
    callbacks: bool = False
    photos: bool = False
    documents: bool = False
    voice: bool = False
    video: bool = False
    streaming: bool = False
    typing_indicator: bool = False
    replies: bool = True
    buttons_text_fallback: bool = True

    def supports(self, feature: str) -> bool:
        """Check if a named feature is supported by this platform."""
        return bool(getattr(self, feature, False))

    def to_dict(self) -> Dict[str, Any]:
        """Convert capability set to a clean dictionary."""
        return asdict(self)
