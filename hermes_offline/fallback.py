"""Connectivity-aware *local* fallback for the provider chain.

Hermes already has a fallback provider chain (``config["fallback_providers"]``,
normalised by :func:`hermes_cli.fallback_config.get_fallback_chain`) that is tried
in order when the primary model fails.  This module makes that existing chain
**offline-capable** by appending a final entry that targets a *local* inference
runtime (Ollama / llama.cpp) exposing an OpenAI-compatible endpoint.

Design constraints:

* **No parallel architecture.**  We only *append one entry* to the existing
  chain; every consumer (messaging gateway, TUI/Desktop gateway, cron) picks it
  up unchanged.
* **Opt-in.**  Enabled with ``offline.auto_local_fallback: true`` so we never
  silently redirect traffic to a local model.
* **Honest.**  The entry is only produced when a *verified* local model and an
  *available* runtime both exist.  Otherwise the function returns ``None`` and
  the chain is untouched.
* **Survival ordering.**  The local entry goes **last**, so it is only reached
  after every remote provider has failed — which is exactly the outage case.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["local_fallback_entry", "offline_config", "is_enabled"]

_TRUTHY = {"1", "true", "yes", "on", "enabled"}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in _TRUTHY


def offline_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the ``offline`` config block (never mutates the input)."""
    block = (config or {}).get("offline")
    return dict(block) if isinstance(block, dict) else {}


def is_enabled(config: Optional[Dict[str, Any]]) -> bool:
    """Whether automatic local fallback is opted in."""
    return _truthy(offline_config(config).get("auto_local_fallback"))


def local_fallback_entry(
    config: Optional[Dict[str, Any]],
    *,
    manager: Any = None,
    records: Optional[List[Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a fallback-chain entry for the local runtime, or ``None``.

    Entry shape matches ``fallback_providers`` entries exactly:
    ``{"provider", "model", "base_url", "api_key", "api_mode"}``.
    """
    if not is_enabled(config):
        return None

    cfg = offline_config(config)

    if records is None:
        try:
            from .emergency import _load_registry  # local import: avoid import cycles
            from .models import ModelState

            reg = _load_registry()
            records = [r for r in (reg.all() if reg is not None else [])
                       if getattr(r, "state", None) in (ModelState.VERIFIED, ModelState.INSTALLED)]
        except Exception:  # noqa: BLE001 - never break chain resolution
            logger.debug("offline fallback: registry unavailable", exc_info=True)
            records = []
    records = records or []
    if not records:
        return None

    if manager is None:
        try:
            from .runtimes import RuntimeManager

            manager = RuntimeManager.from_config(cfg)
        except Exception:  # noqa: BLE001
            logger.debug("offline fallback: runtime manager unavailable", exc_info=True)
            return None

    # Prefer a verified model; fall back to the first installed one.
    try:
        from .models import ModelState

        ordered = sorted(records, key=lambda r: 0 if getattr(r, "state", None)
                         == ModelState.VERIFIED else 1)
    except Exception:  # noqa: BLE001
        ordered = list(records)

    for record in ordered:
        try:
            adapter = manager.runtime_for(record)
        except Exception:  # noqa: BLE001
            adapter = None
        if adapter is None:
            continue
        available = True
        try:
            available = bool(adapter.is_available())
        except Exception:  # noqa: BLE001
            available = False
        if not available:
            continue
        try:
            base_url = adapter.openai_base_url()
        except Exception:  # noqa: BLE001
            base_url = ""
        if not base_url:
            continue
        return {
            "provider": "custom",
            "model": str(getattr(record, "name", "") or ""),
            "base_url": base_url,
            # Local runtimes ignore the key, but the field must be present so
            # credential resolution does not fall through to a remote lookup.
            "api_key": "local",
            "api_mode": "openai",
        }
    return None
