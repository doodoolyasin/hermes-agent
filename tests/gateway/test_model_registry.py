import pytest
from gateway.model_registry import (
    ModelRegistry,
    ModelMetadata,
    ModelHealthStatus,
    ModelHealthProbe,
    HealthProbeResult,
)


def test_model_metadata_capabilities():
    m = ModelMetadata(
        id="test-coder",
        provider="local",
        capabilities=["coding", "tools"],
        context_window=32000,
    )
    assert m.supports("coding") is True
    assert m.supports("tools") is True
    assert m.supports("vision") is False


def test_model_registry_filtering():
    reg = ModelRegistry()
    m_vis = ModelMetadata(
        id="vision-model-1",
        provider="test",
        capabilities=["vision", "chat"],
    )
    reg.register(m_vis)

    vision_models = reg.list_by_capability("vision")
    assert any(m.id == "vision-model-1" for m in vision_models)
    assert not any(m.id == "openai-fast" for m in vision_models)

    coding_models = reg.list_by_capability("coding")
    assert any(m.id == "openai-fast" for m in coding_models)


def test_model_health_probe_mocking():
    class FakeProbe(ModelHealthProbe):
        def probe_endpoint(self, base_url: str, api_key=None):
            if "good-endpoint" in base_url:
                return HealthProbeResult(available=True, latency_ms=120.5)
            return HealthProbeResult(available=False, latency_ms=3000.0, error="Timeout")

    reg = ModelRegistry(probe=FakeProbe())
    reg.register(
        ModelMetadata(id="m-good", provider="p", base_url="http://good-endpoint/v1")
    )
    reg.register(
        ModelMetadata(id="m-bad", provider="p", base_url="http://bad-endpoint/v1")
    )

    st_good = reg.check_health("m-good")
    assert st_good == ModelHealthStatus.HEALTHY
    assert reg.get("m-good").latency_ms == 120.5

    st_bad = reg.check_health("m-bad")
    assert st_bad == ModelHealthStatus.UNHEALTHY
    assert reg.get("m-bad").last_error == "Timeout"
