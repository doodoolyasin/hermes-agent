"""Regression tests: `prepare` must PERSIST the registry.

Before the fix, ``prepare`` added the downloaded artifact to the in-memory
registry but never wrote it to disk, so ``verify_pack`` / ``readiness`` (which
reload the registry from disk) saw nothing: ``checked == 0`` and tampering went
undetected.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from hermes_offline import emergency
from hermes_offline.models import DownloadState, ModelState

PAYLOAD = b"GGUF-weights" * 200
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    return tmp_path / "hh"


class _FakeResult:
    def __init__(self, path, sha):
        self.state = DownloadState.COMPLETE
        self.path = path
        self.sha256 = sha
        self.error = ""


def _install_fake_downloader(monkeypatch):
    class _FakeDownloader:
        def download(self, spec):  # noqa: ANN001
            dest = os.path.join(emergency.models_dir(), spec.filename)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(PAYLOAD)
            return _FakeResult(dest, DIGEST)

    monkeypatch.setattr(emergency, "_load_downloader", lambda **kw: _FakeDownloader())


def test_prepare_persists_registry_so_verify_pack_sees_artifact(home, monkeypatch):
    _install_fake_downloader(monkeypatch)

    report = emergency.prepare([{
        "name": "m1", "url": "http://example.invalid/m1.gguf",
        "filename": "m1.gguf", "sha256": DIGEST,
    }])
    assert "m1" in report.downloaded
    assert report.failed == []

    # verify_pack reloads the registry from disk — it must now see 1 artifact.
    result = emergency.verify_pack()
    assert result["checked"] == 1, "prepare did not persist the artifact"
    assert result["ok"] is True
    assert result["failures"] == []


def test_prepare_persists_across_fresh_registry_load(home, monkeypatch):
    _install_fake_downloader(monkeypatch)
    emergency.prepare([{"name": "m2", "url": "http://example.invalid/m2.gguf",
                        "filename": "m2.gguf", "sha256": DIGEST}])

    # A brand-new readiness() call (fresh registry load) must see the model.
    item = next(i for i in emergency.readiness().items if i.name == "local_model")
    assert item.level.value in ("ready", "degraded")
    assert "m2" in item.detail or "1 " in item.detail


def test_tampering_is_detected_after_prepare(home, monkeypatch):
    _install_fake_downloader(monkeypatch)
    emergency.prepare([{"name": "m3", "url": "http://example.invalid/m3.gguf",
                        "filename": "m3.gguf", "sha256": DIGEST}])
    assert emergency.verify_pack()["ok"] is True

    with open(os.path.join(emergency.models_dir(), "m3.gguf"), "ab") as fh:
        fh.write(b"TAMPERED")

    result = emergency.verify_pack()
    assert result["ok"] is False
    assert result["failures"], "tampering was not detected"


def test_prepare_marks_verified_only_when_checksum_given(home, monkeypatch):
    _install_fake_downloader(monkeypatch)
    # No sha256 supplied -> INSTALLED (not VERIFIED).
    emergency.prepare([{"name": "m4", "url": "http://example.invalid/m4.gguf",
                        "filename": "m4.gguf"}])
    reg = emergency._load_registry()
    record = next(r for r in reg.all() if r.name == "m4")
    assert record.state is ModelState.INSTALLED
