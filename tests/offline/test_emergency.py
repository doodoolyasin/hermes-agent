"""Tests for hermes_offline.emergency (readiness, verify, import, recovery)."""

from __future__ import annotations

import hashlib
import os

import pytest

from hermes_offline import emergency
from hermes_offline.models import ModelState

EXPECTED_ITEMS = {
    "local_model",
    "runtime",
    "tools",
    "memory",
    "offline_docs",
    "recovery",
    "disk",
    "dependencies",
}


@pytest.fixture()
def home(tmp_path, monkeypatch):
    target = tmp_path / "hh"
    monkeypatch.setenv("HERMES_HOME", str(target))
    return target


def test_readiness_reports_all_sections(home):
    rep = emergency.readiness()
    names = {i.name for i in rep.items}
    assert EXPECTED_ITEMS <= names
    assert isinstance(rep.ready, bool)
    # every item carries a level and the report renders without error
    assert all(i.level is not None for i in rep.items)
    assert isinstance(rep.render(), str)


def test_readiness_not_ready_without_local_model(home):
    rep = emergency.readiness()
    local = next(i for i in rep.items if i.name == "local_model")
    assert local.level.value in {"not_ready", "degraded"}
    assert rep.ready is False


def test_checkpoint_roundtrip_and_atomicity(home):
    path = emergency.save_checkpoint("task/1", {"step": 3})
    assert os.path.exists(path)
    loaded = emergency.load_checkpoint("task/1")
    assert loaded["state"]["step"] == 3
    assert emergency.load_checkpoint("does-not-exist") is None
    # no temp file left behind after atomic replace
    assert not os.path.exists(path + ".tmp")


def test_import_model_never_verifies_without_matching_sha(home, tmp_path):
    src = tmp_path / "weights.gguf"
    data = b"weights" * 4096
    src.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()

    rec = emergency.import_model(str(src))
    assert rec.state == ModelState.INSTALLED  # no expected hash supplied
    assert rec.sha256 == digest

    rec_ok = emergency.import_model(str(src), sha256=digest, name="verified-one")
    assert rec_ok.state == ModelState.VERIFIED

    rec_bad = emergency.import_model(str(src), sha256="0" * 64, name="bad-one")
    assert rec_bad.state == ModelState.INSTALLED  # mismatch must never verify


def test_import_missing_file_raises(home):
    with pytest.raises(FileNotFoundError):
        emergency.import_model("/definitely/not/here.gguf")


def test_verify_pack_flags_missing_artifact(home, tmp_path):
    src = tmp_path / "m.gguf"
    src.write_bytes(b"x" * 32)
    digest = hashlib.sha256(b"x" * 32).hexdigest()
    emergency.import_model(str(src), sha256=digest, name="v")
    os.remove(os.path.join(emergency.models_dir(), "m.gguf"))
    result = emergency.verify_pack()
    assert result["ok"] is False
    assert any(f.get("error") == "file missing" for f in result["failures"])


def test_status_shape(home):
    s = emergency.status()
    assert {"ready", "connectivity", "readiness", "paths"} <= set(s)
    assert {"home", "models", "docs", "checkpoints"} <= set(s["paths"])


def test_dependency_snapshot_records_python(home):
    snap = emergency.dependency_snapshot()
    assert "python" in snap and snap["python"]
    assert "packages" in snap
