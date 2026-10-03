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

    def list_supported(self) -> list[str]:
        """Return list of supported feature names."""
        return [k for k, v in self.to_dict().items() if v is True]

    def list_unsupported(self) -> list[str]:
        """Return list of unsupported feature names."""
        return [k for k, v in self.to_dict().items() if v is False]

    def get_fallback_strategy(self, feature: str) -> str:
        """Return recommended fallback strategy when a feature is unsupported."""
        if self.supports(feature):
            return "native"
        if feature in ("inline_keyboard", "callbacks"):
            return "numbered_text_menu" if self.buttons_text_fallback else "omit"
        if feature == "markdown":
            return "plain_text_strip"
        if feature in ("voice", "video"):
            return "text_transcription_or_link"
        return "unsupported"

    def to_dict(self) -> Dict[str, Any]:
        """Convert capability set to a clean dictionary."""
        return asdict(self)
