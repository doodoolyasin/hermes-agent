import pytest
from unittest.mock import MagicMock
from gateway.model_registry import (
    ModelRegistry,
    ModelMetadata,
    ModelHealthStatus,
    ModelHealthProbe,
    HealthProbeResult,
)


def test_probe_high_latency_jitter_transitions_to_degraded():
    mock_probe = MagicMock(spec=ModelHealthProbe)
    # Simulate high latency (5200 ms) due to network jitter
    mock_probe.probe_endpoint.return_value = HealthProbeResult(
        available=True,
        latency_ms=5200.0,
        status_code=200,
    )

    reg = ModelRegistry(probe=mock_probe)
    model = ModelMetadata(
        id="test-jitter-model",
        provider="remote",
        capabilities=["coding"],
        base_url="https://api.test.ai",
    )
    reg.register(model)

    status = reg.check_health("test-jitter-model")
    assert status == ModelHealthStatus.DEGRADED
    assert model.latency_ms == 5200.0
    assert model.last_error is None


def test_probe_fast_response_transitions_to_healthy():
    mock_probe = MagicMock(spec=ModelHealthProbe)
    mock_probe.probe_endpoint.return_value = HealthProbeResult(
        available=True,
        latency_ms=180.0,
        status_code=200,
    )

    reg = ModelRegistry(probe=mock_probe)
    model = ModelMetadata(
        id="test-fast-model",
        provider="local",
        capabilities=["chat"],
        base_url="http://127.0.0.1:11434",
    )
    reg.register(model)

    status = reg.check_health("test-fast-model")
    assert status == ModelHealthStatus.HEALTHY
    assert model.latency_ms == 180.0


def test_probe_connection_refused_transitions_to_unhealthy():
    mock_probe = MagicMock(spec=ModelHealthProbe)
    mock_probe.probe_endpoint.return_value = HealthProbeResult(
        available=False,
        latency_ms=50.0,
        error="Connection refused (Errno 111)",
    )

    reg = ModelRegistry(probe=mock_probe)
    model = ModelMetadata(
        id="test-dead-model",
        provider="local",
        capabilities=["chat"],
        base_url="http://127.0.0.1:9999",
    )
    reg.register(model)

    status = reg.check_health("test-dead-model")
    assert status == ModelHealthStatus.UNHEALTHY
    assert "Connection refused" in model.last_error
