import pytest
from gateway.connectivity_matrix import (
    ConnectivityMatrix,
    ConnectivityStatus,
    LayerProbeResult,
    MatrixReport,
)


def test_layer_probe_result_serialization():
    res = LayerProbeResult("Bale", ConnectivityStatus.ONLINE, latency_ms=45.2)
    d = res.to_dict()
    assert d["layer"] == "Bale"
    assert d["status"] == "ONLINE"
    assert d["latency_ms"] == 45.2


def test_degraded_classification_when_international_blocked():
    matrix = ConnectivityMatrix()

    # Mock specific probes: local & national platforms online, international blocked
    matrix.probe_dns = lambda: LayerProbeResult("DNS", ConnectivityStatus.ONLINE, 10.0)
    matrix.probe_ipv4 = lambda: LayerProbeResult("IPv4", ConnectivityStatus.ONLINE, 15.0)
    matrix.probe_ipv6 = lambda: LayerProbeResult("IPv6", ConnectivityStatus.OFFLINE, error="no route")
    matrix.probe_https = lambda: LayerProbeResult("HTTPS", ConnectivityStatus.ONLINE, 25.0)

    def fake_endpoint(name, url):
        if name in ("Bale", "Rubika"):
            return LayerProbeResult(name, ConnectivityStatus.ONLINE, 20.0)
        if name == "International":
            return LayerProbeResult(name, ConnectivityStatus.OFFLINE, error="Timeout")
        return LayerProbeResult(name, ConnectivityStatus.ONLINE, 100.0)

    matrix.probe_endpoint = fake_endpoint

    report = matrix.evaluate(check_international=True, check_bale=True, check_rubika=True)
    # Must NOT be marked globally OFFLINE because national services and base IP are online!
    assert report.overall_status == ConnectivityStatus.DEGRADED
    assert report.layers["Bale"].status == ConnectivityStatus.ONLINE
    assert report.layers["International"].status == ConnectivityStatus.OFFLINE


def test_total_offline_when_base_unreachable():
    matrix = ConnectivityMatrix()

    matrix.probe_dns = lambda: LayerProbeResult("DNS", ConnectivityStatus.OFFLINE, error="down")
    matrix.probe_ipv4 = lambda: LayerProbeResult("IPv4", ConnectivityStatus.OFFLINE, error="down")
    matrix.probe_ipv6 = lambda: LayerProbeResult("IPv6", ConnectivityStatus.OFFLINE, error="down")
    matrix.probe_https = lambda: LayerProbeResult("HTTPS", ConnectivityStatus.OFFLINE, error="down")
    matrix.probe_endpoint = lambda name, url: LayerProbeResult(name, ConnectivityStatus.OFFLINE, error="down")

    report = matrix.evaluate()
    assert report.overall_status == ConnectivityStatus.OFFLINE
