"""Tests for hermes_offline.download (resumable, checksum-verified downloads).

A real in-process HTTP server with Range support is used so resume, retry and
failover behaviour is exercised for real (no mocks of ``urlopen``).
"""

from __future__ import annotations

import hashlib
import os
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_offline import download
from hermes_offline.models import DownloadState

PAYLOAD = bytes(range(256)) * 200  # 51200 bytes, non-trivial + deterministic
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class _RangeHandler(BaseHTTPRequestHandler):
    payload = b""
    supports_range = True
    fail_status = 0
    fail_times = 0
    hits: list = []
    lock = threading.Lock()

    def log_message(self, *args):  # keep test output clean
        return

    def do_GET(self):
        cls = type(self)
        with cls.lock:
            cls.hits.append(self.headers.get("Range"))
            if cls.fail_times > 0:
                cls.fail_times -= 1
                send_fail = True
            else:
                send_fail = False
        if send_fail:
            self.send_response(cls.fail_status or 503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        rng = self.headers.get("Range")
        if rng and cls.supports_range:
            try:
                start = int(rng.split("=", 1)[1].split("-", 1)[0])
            except (IndexError, ValueError):
                start = 0
            body = cls.payload[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(cls.payload) - 1}/{len(cls.payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Length", str(len(cls.payload)))
        self.send_header("Accept-Ranges", "none")
        self.end_headers()
        self.wfile.write(cls.payload)


@pytest.fixture()
def server():
    _RangeHandler.payload = PAYLOAD
    _RangeHandler.supports_range = True
    _RangeHandler.fail_status = 0
    _RangeHandler.fail_times = 0
    _RangeHandler.hits = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield types.SimpleNamespace(httpd=httpd, base=base, handler=_RangeHandler, url=f"{base}/model.bin")
    finally:
        httpd.shutdown()
        httpd.server_close()


def _spec(url, dest_dir, *, sha=SHA, **kw):
    return download.DownloadSpec(
        name="model.bin",
        url=url,
        sha256=sha,
        dest_dir=str(dest_dir),
        filename="model.bin",
        size_bytes=kw.pop("size_bytes", len(PAYLOAD)),
        **kw,
    )


# -- local / file:// transfer ---------------------------------------------


def test_file_url_download_verifies_and_installs(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(PAYLOAD)
    spec = _spec("file://" + str(src), tmp_path / "dl")
    result = download.Downloader(backoff=0).download(spec)
    assert result.state == DownloadState.COMPLETE
    assert result.sha256 == SHA
    assert os.path.exists(result.path)
    assert not os.path.exists(result.part_path)  # atomic rename, no partial left
    assert open(result.path, "rb").read() == PAYLOAD


def test_bare_path_source_is_supported(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(PAYLOAD)
    result = download.Downloader(backoff=0).download(_spec(str(src), tmp_path / "dl"))
    assert result.state == DownloadState.COMPLETE


def test_checksum_mismatch_quarantines_and_never_installs(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(PAYLOAD)
    spec = _spec("file://" + str(src), tmp_path / "dl", sha="0" * 64)
    result = download.Downloader(backoff=0).download(spec)
    assert result.state == DownloadState.CHECKSUM_MISMATCH
    assert not os.path.exists(result.path)
    assert result.quarantine_path and os.path.exists(result.quarantine_path)
    assert not os.path.exists(result.part_path)


def test_missing_checksum_still_installs_and_reports_digest(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(PAYLOAD)
    result = download.Downloader(backoff=0).download(_spec("file://" + str(src), tmp_path / "dl", sha=""))
    assert result.state == DownloadState.COMPLETE
    assert result.sha256 == SHA  # computed locally even without an expected hash


# -- HTTP transfer ---------------------------------------------------------


def test_http_download_complete(server, tmp_path):
    result = download.Downloader(backoff=0).download(_spec(server.url, tmp_path))
    assert result.state == DownloadState.COMPLETE
    assert open(result.path, "rb").read() == PAYLOAD


def test_http_resume_uses_range_and_appends(server, tmp_path):
    dest = tmp_path / "model.bin"
    part = tmp_path / "model.bin.part"
    part.write_bytes(PAYLOAD[:500])  # simulate an interrupted transfer
    result = download.Downloader(backoff=0).download(_spec(server.url, tmp_path))
    assert result.state == DownloadState.COMPLETE
    assert result.resumed is True
    assert open(dest, "rb").read() == PAYLOAD
    assert any(r and r.startswith("bytes=500-") for r in server.handler.hits)


def test_server_ignoring_range_restarts_cleanly(server, tmp_path):
    server.handler.supports_range = False
    (tmp_path / "model.bin.part").write_bytes(PAYLOAD[:300])
    result = download.Downloader(backoff=0).download(_spec(server.url, tmp_path))
    assert result.state == DownloadState.COMPLETE
    assert result.resumed is False
    assert open(result.path, "rb").read() == PAYLOAD


def test_transient_5xx_is_retried(server, tmp_path):
    server.handler.fail_times = 1
    server.handler.fail_status = 503
    result = download.Downloader(backoff=0, retries=3).download(_spec(server.url, tmp_path))
    assert result.state == DownloadState.COMPLETE
    assert result.attempts >= 2


def test_permanent_4xx_fails_without_endless_retry(server, tmp_path):
    server.handler.fail_times = 99
    server.handler.fail_status = 404
    result = download.Downloader(backoff=0, retries=3).download(_spec(server.url, tmp_path))
    assert result.state == DownloadState.FAILED
    assert "404" in result.error
    assert result.attempts == 1  # 4xx is not retried


def test_source_failover_to_mirror(server, tmp_path):
    dead = "http://127.0.0.1:1/model.bin"  # connection refused
    spec = download.DownloadSpec(
        name="model.bin", url=dead, sources=[server.url], sha256=SHA,
        dest_dir=str(tmp_path), filename="model.bin", size_bytes=len(PAYLOAD),
    )
    result = download.Downloader(backoff=0, retries=1).download(spec)
    assert result.state == DownloadState.COMPLETE
    assert result.source == server.url


def test_mirror_base_url_is_resolved(server, tmp_path):
    dead = "http://127.0.0.1:1/model.bin"
    spec = download.DownloadSpec(
        name="model.bin", url=dead, sha256=SHA,
        dest_dir=str(tmp_path), filename="model.bin", size_bytes=len(PAYLOAD),
    )
    dl = download.Downloader(backoff=0, retries=1, mirrors=[server.base])
    result = dl.download(spec)
    assert result.state == DownloadState.COMPLETE


def test_offline_mode_rejects_http(server, tmp_path):
    result = download.Downloader(backoff=0, allow_network=False).download(_spec(server.url, tmp_path))
    assert result.state == DownloadState.FAILED
    assert "network" in result.error.lower()


def test_idempotent_when_destination_already_correct(server, tmp_path):
    dl = download.Downloader(backoff=0)
    first = dl.download(_spec(server.url, tmp_path))
    assert first.state == DownloadState.COMPLETE
    before = len(server.handler.hits)
    second = dl.download(_spec(server.url, tmp_path))
    assert second.state == DownloadState.COMPLETE
    assert len(server.handler.hits) == before  # no re-download


# -- preflight / helpers ---------------------------------------------------


def test_insufficient_disk_is_reported(monkeypatch, tmp_path):
    class _Usage:
        total = 10
        used = 10
        free = 1  # 1 byte free

    monkeypatch.setattr(download.shutil, "disk_usage", lambda _p: _Usage)
    src = tmp_path / "src.bin"
    src.write_bytes(PAYLOAD)
    result = download.Downloader(backoff=0).download(_spec("file://" + str(src), tmp_path / "dl"))
    assert result.state == DownloadState.INSUFFICIENT_DISK
    assert not os.path.exists(result.path)


def test_verify_file_and_sha256_file(tmp_path):
    artifact = tmp_path / "a.bin"
    artifact.write_bytes(b"abc")
    assert download.sha256_file(str(artifact)) == hashlib.sha256(b"abc").hexdigest()
    assert download.verify_file(str(artifact)) == hashlib.sha256(b"abc").hexdigest()
    assert download.verify_file(str(tmp_path / "missing.bin")) is None
    with pytest.raises(OSError):
        download.sha256_file(str(tmp_path / "missing.bin"))


def test_progress_callback_receives_totals(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(PAYLOAD)
    seen = []
    dl = download.Downloader(backoff=0, chunk_size=4096, progress=lambda done, total: seen.append((done, total)))
    result = dl.download(_spec("file://" + str(src), tmp_path / "dl"))
    assert result.state == DownloadState.COMPLETE
    assert seen and seen[-1][0] == len(PAYLOAD)
    assert seen[-1][1] == len(PAYLOAD)


def test_spec_source_dedup_and_filename_resolution(tmp_path):
    spec = download.DownloadSpec(
        name="", url="https://host/path/thing.gguf", sources=["https://host/path/thing.gguf", "https://mirror/x"],
    )
    assert spec.all_sources() == ["https://host/path/thing.gguf", "https://mirror/x"]
    assert spec.resolved_filename() == "thing.gguf"


def test_no_source_reports_failure(tmp_path):
    spec = download.DownloadSpec(name="x", url="", dest_dir=str(tmp_path), filename="x.bin")
    result = download.Downloader(backoff=0).download(spec)
    assert result.state == DownloadState.FAILED
    assert "source" in result.error.lower()


def test_cleanup_partials(tmp_path):
    (tmp_path / "a.part").write_bytes(b"x")
    (tmp_path / "b.bin").write_bytes(b"y")
    removed = download.Downloader(dest_dir=str(tmp_path)).cleanup_partials()
    assert [os.path.basename(p) for p in removed] == ["a.part"]
    assert (tmp_path / "b.bin").exists()
