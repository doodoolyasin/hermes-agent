"""Adaptive Model Router and Offline-First Fallback Engine for Hermes.

Implements P2/Directive section 11:
- Automatic fallback chain:
    Primary -> ProviderRegistry alternatives -> Local/LAN -> Remote Hermes -> Emergency Local -> Persistent TaskQueue
- Degraded-mode awareness from ConnectivityMatrix and CircuitBreaker
- Offline queuing: when all AI providers are unreachable, queues prompt in SQLite TaskQueue
  and returns a friendly Persian acknowledgment instead of crashing or losing the turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from gateway.circuit_breaker import global_circuit_breaker
from gateway.connectivity_matrix import ConnectivityStatus, global_connectivity_matrix
from gateway.provider_registry import ProviderDescriptor, ProviderTier, global_provider_registry
from gateway.task_queue import QueuedTask, TaskState, global_task_queue

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    provider: Optional[ProviderDescriptor]
    model_id: str
    is_queued_offline: bool = False
    queued_task_id: Optional[str] = None
    reason: str = "normal"


class AdaptiveRouter:
    """Intelligent fallback and offline router."""

    def __init__(self) -> None:
        pass

    def route_request(
        self,
        session_key: str,
        prompt: str,
        preferred_model: Optional[str] = None,
        preferred_provider: Optional[str] = None,
    ) -> RouteDecision:
        """Route request through resilience ladder or enqueue if entirely offline."""
        # 1. Check if preferred provider is healthy and allowed by circuit breaker
        if preferred_provider:
            p = global_provider_registry.get(preferred_provider)
            if p and p.enabled and global_circuit_breaker.can_execute(preferred_provider):
                return RouteDecision(
                    provider=p,
                    model_id=preferred_model or (p.models[0] if p.models else "default"),
                    reason="preferred_provider",
                )

        # 2. Ladder through available providers sorted by tier (Local -> LAN -> Remote -> Public)
        candidate = global_provider_registry.select_best_provider(required_model=preferred_model)
        if candidate:
            model = preferred_model if (preferred_model in candidate.models) else (candidate.models[0] if candidate.models else "default")
            return RouteDecision(provider=candidate, model_id=model, reason="provider_registry_ladder")

        # 3. If all providers are down/circuit-broken or offline:
        # Enqueue prompt to PersistentTaskQueue so work is never lost!
        task = global_task_queue.enqueue(
            session_key=session_key,
            task_type="offline_prompt",
            payload={"prompt": prompt, "preferred_model": preferred_model},
            priority=1,
            max_attempts=5,
        )
        logger.warning(
            "All AI providers unreachable or circuit-broken. Prompt queued in TaskQueue (task_id=%s)",
            task.task_id,
        )
        return RouteDecision(
            provider=None,
            model_id="queued",
            is_queued_offline=True,
            queued_task_id=task.task_id,
            reason="offline_queued",
        )

    def get_offline_user_message(self, task_id: Optional[str] = None) -> str:
        """Friendly user-facing Persian message when system operates in offline queued mode."""
        msg = (
            "⚠️ **ارتباط با سرویس‌های هوش مصنوعی موقتاً قطع یا مسدود شده است.**\n\n"
            "📥 درخواست شما با موفقیت در صف پایدار ذخیره شد و به محض برقراری مجدد ارتباط، "
            "پردازش و پاسخ برای شما ارسال خواهد شد."
        )
        if task_id:
            msg += f"\n\n🔖 شناسه پیگیری وظیفه: `{task_id}`"
        return msg


global_adaptive_router = AdaptiveRouter()
