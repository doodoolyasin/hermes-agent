import pytest
from gateway.retry_engine import RetryPolicy


def test_compute_delay_extreme_attempts_no_overflow():
    policy = RetryPolicy(
        base_delay=1.0,
        max_delay=60.0,
        factor=2.0,
        jitter="none",
    )

    # 1. Normal attempt
    assert policy.compute_delay(1) == 1.0
    assert policy.compute_delay(2) == 2.0
    assert policy.compute_delay(3) == 4.0
    assert policy.compute_delay(10) == 60.0

    # 2. Extreme attempt count that would previously cause OverflowError (e.g. 2**1200)
    delay_extreme = policy.compute_delay(1200)
    assert delay_extreme == 60.0

    # 3. Negative or zero attempt
    assert policy.compute_delay(0) == 1.0
    assert policy.compute_delay(-10) == 1.0


def test_compute_delay_full_and_equal_jitter_bounds():
    policy_full = RetryPolicy(
        base_delay=2.0,
        max_delay=10.0,
        jitter="full",
    )
    for _ in range(50):
        d = policy_full.compute_delay(5)
        assert 0.0 <= d <= 10.0

    policy_equal = RetryPolicy(
        base_delay=2.0,
        max_delay=10.0,
        jitter="equal",
    )
    for _ in range(50):
        d = policy_equal.compute_delay(5)
        # equal jitter: half <= d <= max_delay
        assert 5.0 <= d <= 10.0
