import socket
import ssl
import pytest
from unittest.mock import patch, MagicMock
from gateway.connectivity_matrix import (
    ConnectivityMatrix,
    ConnectivityStatus,
)


def test_matrix_os_socket_errors_handled_gracefully():
    matrix = ConnectivityMatrix(probe_timeout=0.1)

    # 1. DNS raises socket.gaierror (e.g. [Errno -3] Temporary failure in name resolution)
    with patch("socket.getaddrinfo", side_effect=socket.gaierror(-3, "Temporary failure")):
        res = matrix.probe_dns()
        assert res.status == ConnectivityStatus.OFFLINE
        assert "Temporary failure" in res.error

    # 2. IPv4 raises OSError (e.g. [Errno 101] Network is unreachable)
    with patch.object(socket.socket, "connect", side_effect=OSError(101, "Network is unreachable")):
        res = matrix.probe_ipv4()
        assert res.status == ConnectivityStatus.OFFLINE
        assert "Network is unreachable" in res.error

    # 3. HTTPS raises SSLError (e.g. handshake failure)
    with patch("socket.create_connection", return_value=MagicMock()):
        with patch.object(ssl.SSLContext, "wrap_socket", side_effect=ssl.SSLError("SSL Handshake failed")):
            res = matrix.probe_https()
            assert res.status == ConnectivityStatus.OFFLINE
            assert "SSL Handshake failed" in res.error


def test_matrix_partial_subnet_blackout_evaluation():
    matrix = ConnectivityMatrix()

    # Mock evaluate layers: DNS, IPv4, Bale and Rubika are ONLINE,
    # but International is OFFLINE (e.g. national intranet / filtering active)
    with patch.object(matrix, "probe_dns", return_value=MagicMock(status=ConnectivityStatus.ONLINE)), \
         patch.object(matrix, "probe_ipv4", return_value=MagicMock(status=ConnectivityStatus.ONLINE)), \
         patch.object(matrix, "probe_ipv6", return_value=MagicMock(status=ConnectivityStatus.OFFLINE)), \
         patch.object(matrix, "probe_https", return_value=MagicMock(status=ConnectivityStatus.ONLINE)):

        def mock_probe_endpoint(name, url):
            if name in ("Bale", "Rubika"):
                return MagicMock(status=ConnectivityStatus.ONLINE)
            return MagicMock(status=ConnectivityStatus.OFFLINE, error="Timeout")

        with patch.object(matrix, "probe_endpoint", side_effect=mock_probe_endpoint):
            report = matrix.evaluate()
            # Because base is online and local messengers are online, but International is down,
            # overall must be DEGRADED (operational for local traffic), never OFFLINE!
            assert report.overall_status == ConnectivityStatus.DEGRADED
            assert report.layers["Bale"].status == ConnectivityStatus.ONLINE
            assert report.layers["Rubika"].status == ConnectivityStatus.ONLINE
            assert report.layers["International"].status == ConnectivityStatus.OFFLINE
