"""Central Retry Engine for Hermes.

Implements P1-B4 and specification section 18:
- Centralized, configurable RetryPolicy with full/equal/no jitter
- Exponential backoff with bounded max_delay
- Classification of retryable vs non-retryable exceptions and HTTP status codes
- Deep integration with ProviderCircuitBreaker
- Telemetry/hook notification on each retry attempt
- Both async and sync execution APIs and decorator
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from gateway.circuit_breaker import ProviderCircuitBreaker

logger = logging.getLogger(__name__)
T = TypeVar("T")

DEFAULT_RETRYABLE_STATUS_CODES: Set[int] = {408, 429, 500, 502, 503, 504}


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    factor: float = 2.0
    jitter: str = "full"  # 'full', 'equal', 'none'
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (
        KeyboardInterrupt,
        SystemExit,
        asyncio.CancelledError,
    )
    retryable_status_codes: Set[int] = field(default_factory=lambda: set(DEFAULT_RETRYABLE_STATUS_CODES))
    circuit_breaker: Optional[ProviderCircuitBreaker] = None
    provider_name: Optional[str] = None
    on_retry: Optional[Callable[[int, Exception, float], None]] = None

    def compute_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay with selected jitter strategy."""
        exp = min(60, max(0, attempt - 1))
        try:
            factor_val = self.factor ** exp
            raw_delay = min(self.max_delay, self.base_delay * factor_val)
        except OverflowError:
            raw_delay = self.max_delay

        if self.jitter == "full":
            return random.uniform(0.0, raw_delay)
        elif self.jitter == "equal":
            half = raw_delay / 2.0
            return half + random.uniform(0.0, half)
        return raw_delay

    def is_retryable(self, exc: Exception) -> bool:
        """Check if an exception is considered retryable under this policy."""
        if isinstance(exc, self.non_retryable_exceptions):
            return False

        # Status code checking if available on exception (e.g., status_code or code attribute)
        status_code = getattr(exc, "status_code", getattr(exc, "code", getattr(exc, "status", None)))
        if status_code is not None and isinstance(status_code, int):
            if status_code in (401, 403, 404):
                return False
            if status_code in self.retryable_status_codes:
                return True

        return isinstance(exc, self.retryable_exceptions)


class RetryEngine:
    """Central engine managing retries across Hermes services and platform calls."""

    @staticmethod
    async def execute_async(
        coro_fn: Callable[..., Any],
        policy: RetryPolicy,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute an async coroutine function with retry logic."""
        attempt = 1
        while True:
            # Check circuit breaker if configured
            if policy.circuit_breaker and policy.provider_name:
                if not policy.circuit_breaker.can_execute(policy.provider_name):
                    raise RuntimeError(f"Circuit breaker OPEN for provider '{policy.provider_name}'")

            try:
                result = await coro_fn(*args, **kwargs)
                if policy.circuit_breaker and policy.provider_name:
                    policy.circuit_breaker.record_success(policy.provider_name)
                return result
            except Exception as exc:
                if policy.circuit_breaker and policy.provider_name:
                    policy.circuit_breaker.record_failure(policy.provider_name, str(exc))

                if attempt >= policy.max_attempts or not policy.is_retryable(exc):
                    raise exc

                delay = policy.compute_delay(attempt)
                logger.warning(
                    "Retryable error on attempt %d/%d: %s. Retrying in %.2fs...",
                    attempt,
                    policy.max_attempts,
                    exc,
                    delay,
                )
                if policy.on_retry:
                    try:
                        policy.on_retry(attempt, exc, delay)
                    except Exception:
                        pass

                await asyncio.sleep(delay)
                attempt += 1

    @staticmethod
    def execute_sync(
        fn: Callable[..., Any],
        policy: RetryPolicy,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a synchronous function with retry logic."""
        attempt = 1
        while True:
            if policy.circuit_breaker and policy.provider_name:
                if not policy.circuit_breaker.can_execute(policy.provider_name):
                    raise RuntimeError(f"Circuit breaker OPEN for provider '{policy.provider_name}'")

            try:
                result = fn(*args, **kwargs)
                if policy.circuit_breaker and policy.provider_name:
                    policy.circuit_breaker.record_success(policy.provider_name)
                return result
            except Exception as exc:
                if policy.circuit_breaker and policy.provider_name:
                    policy.circuit_breaker.record_failure(policy.provider_name, str(exc))

                if attempt >= policy.max_attempts or not policy.is_retryable(exc):
                    raise exc

                delay = policy.compute_delay(attempt)
                logger.warning(
                    "Retryable error on attempt %d/%d: %s. Retrying in %.2fs...",
                    attempt,
                    policy.max_attempts,
                    exc,
                    delay,
                )
                if policy.on_retry:
                    try:
                        policy.on_retry(attempt, exc, delay)
                    except Exception:
                        pass

                time.sleep(delay)
                attempt += 1


def with_retry(policy: Optional[RetryPolicy] = None):
    """Decorator to apply retry logic to sync or async functions."""
    active_policy = policy or RetryPolicy()

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await RetryEngine.execute_async(fn, active_policy, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return RetryEngine.execute_sync(fn, active_policy, *args, **kwargs)
            return sync_wrapper

    return decorator
