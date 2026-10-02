"""Multi-Layer Connectivity Matrix for Hermes.

Implements P1-B2 and architecture sections 16 & 17:
- Multi-layer tracking: DNS, IPv4, IPv6, HTTPS, International, Bale, Rubika, Local AI, Remote AI
- Tri-state classification: ONLINE, DEGRADED, OFFLINE
- Non-binary degradation logic: partial outages trigger fallback routing, not global failure
- Tunable timeouts and bounded probe latencies
"""

from __future__ import annotations

import logging
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConnectivityStatus(str, Enum):
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


@dataclass
class LayerProbeResult:
    layer: str
    status: ConnectivityStatus
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class MatrixReport:
    overall_status: ConnectivityStatus
    layers: Dict[str, LayerProbeResult]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "timestamp": self.timestamp,
        }


class ConnectivityMatrix:
    """Evaluates multi-layered network reachability with fine-grained degradation tracking."""

    def __init__(self, probe_timeout: float = 3.0) -> None:
        self.probe_timeout = probe_timeout
        self._last_report: Optional[MatrixReport] = None

    def probe_dns(self) -> LayerProbeResult:
        start = time.perf_counter()
        try:
            # Query standard root/public host
            socket.getaddrinfo("one.one.one.one", 53, socket.AF_INET, socket.SOCK_STREAM)
            ms = (time.perf_counter() - start) * 1000.0
            return LayerProbeResult("DNS", ConnectivityStatus.ONLINE, round(ms, 1))
        except Exception as exc:
            ms = (time.perf_counter() - start) * 1000.0
            return LayerProbeResult("DNS", ConnectivityStatus.OFFLINE, round(ms, 1), error=str(exc))

    def probe_ipv4(self) -> LayerProbeResult:
        start = time.perf_counter()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.probe_timeout)
        try:
            # Connect to Cloudflare 1.1.1.1 on port 53 (DNS TCP)
            s.connect(("1.1.1.1", 53))
            ms = (time.perf_counter() - start) * 1000.0
            return LayerProbeResult("IPv4", ConnectivityStatus.ONLINE, round(ms, 1))
        except Exception as exc:
            ms = (time.perf_counter() - start) * 1000.0
            return LayerProbeResult("IPv4", ConnectivityStatus.OFFLINE, round(ms, 1), error=str(exc))
        finally:
            s.close()

    def probe_ipv6(self) -> LayerProbeResult:
        if not socket.has_ipv6:
            return LayerProbeResult("IPv6", ConnectivityStatus.OFFLINE, error="IPv6 not supported by host")
        start = time.perf_counter()
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.settimeout(self.probe_timeout)
        try:
            # Google IPv6 DNS (2001:4860:4860::8888)
            s.connect(("2001:4860:4860::8888", 53))
            ms = (time.perf_counter() - start) * 1000.0
            return LayerProbeResult("IPv6", ConnectivityStatus.ONLINE, round(ms, 1))
        except Exception as exc:
            ms = (time.perf_counter() - start) * 1000.0
            # Absence of IPv6 route is treated as degraded/offline for IPv6 layer only, not fatal
            return LayerProbeResult("IPv6", ConnectivityStatus.OFFLINE, round(ms, 1), error=str(exc))
        finally:
            s.close()

    def probe_https(self, host: str = "1.1.1.1", port: int = 443) -> LayerProbeResult:
        start = time.perf_counter()
        try:
            ctx = ssl.create_default_context()
            s = socket.create_connection((host, port), timeout=self.probe_timeout)
            with ctx.wrap_socket(s, server_hostname="cloudflare-dns.com") as ssock:
                ms = (time.perf_counter() - start) * 1000.0
                return LayerProbeResult("HTTPS", ConnectivityStatus.ONLINE, round(ms, 1))
        except Exception as exc:
            ms = (time.perf_counter() - start) * 1000.0
            return LayerProbeResult("HTTPS", ConnectivityStatus.OFFLINE, round(ms, 1), error=str(exc))

    def probe_endpoint(self, name: str, url: str) -> LayerProbeResult:
        start = time.perf_counter()
        req = urllib.request.Request(url, headers={"User-Agent": "Hermes-Connectivity-Probe/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.probe_timeout) as resp:
                ms = (time.perf_counter() - start) * 1000.0
                status = ConnectivityStatus.ONLINE if resp.status < 400 else ConnectivityStatus.DEGRADED
                return LayerProbeResult(name, status, round(ms, 1))
        except urllib.error.HTTPError as exc:
            ms = (time.perf_counter() - start) * 1000.0
            # 404 or 401 proves network reachability to server!
            status = ConnectivityStatus.ONLINE if exc.code in (401, 403, 404, 405) else ConnectivityStatus.DEGRADED
            return LayerProbeResult(name, status, round(ms, 1), error=f"HTTP {exc.code}")
        except Exception as exc:
            ms = (time.perf_counter() - start) * 1000.0
            return LayerProbeResult(name, ConnectivityStatus.OFFLINE, round(ms, 1), error=str(exc))

    def evaluate(
        self,
        *,
        check_international: bool = True,
        check_bale: bool = True,
        check_rubika: bool = True,
        check_local_ai: bool = True,
        check_remote_ai: bool = True,
    ) -> MatrixReport:
        """Evaluate all layers and calculate aggregated system status."""
        layers: Dict[str, LayerProbeResult] = {}

        layers["DNS"] = self.probe_dns()
        layers["IPv4"] = self.probe_ipv4()
        layers["IPv6"] = self.probe_ipv6()
        layers["HTTPS"] = self.probe_https()

        if check_international:
            layers["International"] = self.probe_endpoint("International", "https://www.cloudflare.com/cdn-cgi/trace")

        if check_bale:
            layers["Bale"] = self.probe_endpoint("Bale", "https://tapi.bale.ai")

        if check_rubika:
            layers["Rubika"] = self.probe_endpoint("Rubika", "https://botapi.rubika.ir/v01")

        if check_local_ai:
            layers["Local AI"] = self.probe_endpoint("Local AI", "http://127.0.0.1:11434/api/tags")

        if check_remote_ai:
            layers["Remote AI"] = self.probe_endpoint("Remote AI", "https://text.pollinations.ai/openai/models")

        # Classify overall status
        base_online = any(layers[k].status == ConnectivityStatus.ONLINE for k in ("DNS", "IPv4", "HTTPS") if k in layers)

        if not base_online:
            overall = ConnectivityStatus.OFFLINE
        else:
            # Check if any layer other than IPv6 is not ONLINE
            non_v6_issues = [
                k for k, v in layers.items()
                if k != "IPv6" and v.status != ConnectivityStatus.ONLINE
            ]
            if not non_v6_issues:
                overall = ConnectivityStatus.ONLINE
            else:
                overall = ConnectivityStatus.DEGRADED

        report = MatrixReport(
            overall_status=overall,
            layers=layers,
            timestamp=time.time(),
        )
        self._last_report = report
        return report

    @property
    def last_report(self) -> Optional[MatrixReport]:
        return self._last_report


global_connectivity_matrix = ConnectivityMatrix()
