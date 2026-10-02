"""Tests for hermes_offline.registry (atomic JSON model registry)."""

from __future__ import annotations

import json
import os

import pytest

from hermes_offline import registry
from hermes_offline.models import Channel, ModelRecord, ModelRequirements, ModelState


@pytest.fixture()
def home(tmp_path, monkeypatch):
    target = tmp_path / "hh"
    monkeypatch.setenv("HERMES_HOME", str(target))
    return target


def _rec(name="m1", **kw):
    kw.setdefault("publisher", "acme")
    return ModelRecord(name=name, **kw)


# -- basics ----------------------------------------------------------------


def test_missing_file_is_an_empty_registry(tmp_path):
    reg = registry.ModelRegistry(str(tmp_path / "nope.json"))
    assert reg.all() == []
    assert len(reg) == 0
    assert reg.load_error == ""


def test_default_path_respects_hermes_home(home):
    assert registry.default_registry_path() == os.path.join(str(home), "offline", "registry.json")


def test_add_get_remove_roundtrip(tmp_path):
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    rec = _rec()
    reg.add(rec)
    assert len(reg) == 1
    assert reg.get(rec.key) is rec
    assert reg.get("m1") is rec  # unique-name fallback
    assert rec.key in reg
    assert reg.remove(rec.key) is True
    assert len(reg) == 0
    assert reg.remove("nope") is False


def test_add_duplicate_requires_replace(tmp_path):
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    reg.add(_rec())
    with pytest.raises(ValueError):
        reg.add(_rec())
    reg.add(_rec(quantization="Q8_0"), replace=True)
    assert reg.get("acme/m1").quantization == "Q8_0"


def test_add_discovered_preserves_own_state(tmp_path):
    """The discovery pipeline must not clobber a real install's state."""
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    rec = _rec(state=ModelState.VERIFIED, local_path="/models/m1.gguf")
    reg.add_discovered(rec, replace=True)
    assert reg.get(rec.key).state == ModelState.VERIFIED
    assert reg.get(rec.key).local_path == "/models/m1.gguf"


# -- persistence -----------------------------------------------------------


def test_save_load_roundtrip_preserves_types(tmp_path):
    path = str(tmp_path / "r.json")
    reg = registry.ModelRegistry(path)
    rec = _rec(
        state=ModelState.VERIFIED,
        channel=Channel.EXPERIMENTAL,
        requirements=ModelRequirements(ram_gb=4.0, vram_gb=2.0, context_length=8192),
        capabilities=["chat", "tools"],
        sha256="a" * 64,
        local_path="/models/m1.gguf",
        extra={"runtime": "llama.cpp"},
    )
    reg.add(rec)
    reg.save()

    reg2 = registry.ModelRegistry(path)
    assert len(reg2) == 1
    got = reg2.get(rec.key)
    assert got.state == ModelState.VERIFIED
    assert got.channel == Channel.EXPERIMENTAL
    assert got.requirements.ram_gb == 4.0
    assert got.capabilities == ["chat", "tools"]
    assert got.sha256 == "a" * 64
    assert got.extra["runtime"] == "llama.cpp"
    assert got.discovered_at is not None


def test_save_is_atomic_no_temp_left(tmp_path):
    path = str(tmp_path / "r.json")
    reg = registry.ModelRegistry(path)
    reg.add(_rec())
    reg.save()
    leftovers = [f for f in os.listdir(tmp_path) if ".tmp" in f]
    assert leftovers == []
    assert json.load(open(path, encoding="utf-8"))["version"] == registry.REGISTRY_VERSION


def test_corrupt_registry_is_quarantined_not_fatal(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("{not valid json", encoding="utf-8")
    reg = registry.ModelRegistry(str(path))
    assert reg.all() == []
    assert reg.load_error  # records the problem
    assert reg.quarantined_path
    assert os.path.exists(reg.quarantined_path)
    # The corrupt original was moved aside, so a fresh save succeeds.
    reg.add(_rec())
    reg.save()
    assert registry.ModelRegistry(str(path)).get("acme/m1") is not None


def test_reload_picks_up_external_changes(tmp_path):
    path = str(tmp_path / "r.json")
    reg = registry.ModelRegistry(path)
    reg.add(_rec("one"))
    reg.save()
    other = registry.ModelRegistry(path)
    other.add(_rec("two"))
    other.save()
    reg.reload()
    names = {r.name for r in reg.all()}
    assert {"one", "two"} <= names


# -- lifecycle -------------------------------------------------------------


def test_lifecycle_transitions(tmp_path):
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    rec = _rec()
    reg.add(rec)
    reg.mark_downloading(rec.key, source_url="https://example/model.gguf")
    assert reg.get(rec.key).state == ModelState.DOWNLOADING
    assert reg.get(rec.key).source_url.endswith("model.gguf")

    reg.mark_installed(rec.key, local_path=str(tmp_path / "m1.gguf"), sha256="b" * 64)
    got = reg.get(rec.key)
    assert got.state == ModelState.INSTALLED
    assert got.installed_at is not None

    reg.mark_verified(rec.key)
    assert reg.get(rec.key).state == ModelState.VERIFIED
    assert reg.get(rec.key).last_verified is not None

    reg.mark_failed(rec.key, "boom")
    assert reg.get(rec.key).state == ModelState.FAILED
    assert reg.get(rec.key).fail_count == 1
    assert reg.get(rec.key).last_error == "boom"


def test_mark_verified_requires_local_path(tmp_path):
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    reg.add(_rec())
    with pytest.raises(ValueError):
        reg.mark_verified("acme/m1")


def test_mark_installed_computes_disk_usage(tmp_path):
    artifact = tmp_path / "m1.gguf"
    artifact.write_bytes(b"x" * 2048)
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    reg.add(_rec())
    reg.mark_installed("acme/m1", local_path=str(artifact))
    assert reg.get("acme/m1").disk_usage_gb > 0


def test_by_state_helpers(tmp_path):
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    reg.add(_rec("a", state=ModelState.DISCOVERED))
    reg.add(_rec("b", state=ModelState.INSTALLED))
    reg.add(_rec("c", state=ModelState.VERIFIED, local_path="/c"))
    assert {r.name for r in reg.by_state(ModelState.VERIFIED)} == {"c"}
    assert {r.name for r in reg.installed()} == {"b", "c"}
    assert {r.name for r in reg.verified()} == {"c"}


def test_find_and_disk_total(tmp_path):
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    reg.add(_rec("alpha", quantization="Q4_K_M", disk_usage_gb=1.5))
    reg.add(_rec("beta", publisher="other", disk_usage_gb=2.0))
    assert [r.name for r in reg.find("Q4_K")] == ["alpha"]
    assert len(reg.find("")) == 2
    assert reg.total_disk_usage_gb() == 3.5


# -- maintenance -----------------------------------------------------------


def test_prune_protects_verified_and_pinned(tmp_path):
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    reg.add(_rec("disc", state=ModelState.DISCOVERED))
    reg.add(_rec("keep-verified", state=ModelState.VERIFIED, local_path="/v"))
    reg.add(_rec("keep-pinned", state=ModelState.DISCOVERED, pinned=True))
    removed = reg.prune()
    assert {r.name for r in removed} == {"disc"}
    assert {r.name for r in reg.all()} == {"keep-verified", "keep-pinned"}


def test_prune_spares_pinned_even_without_keep_flag(tmp_path):
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    reg.add(_rec("pinned", state=ModelState.DISCOVERED, pinned=True))
    removed = reg.prune(keep_pinned=False)
    assert {r.name for r in removed} == {"pinned"}


def test_remove_refuses_pinned(tmp_path):
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    reg.add(_rec("p", pinned=True))
    assert reg.remove("acme/p") is False
    assert reg.remove("acme/p", allow_pinned=True) is True


def test_prune_dry_run_does_not_mutate(tmp_path):
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    reg.add(_rec("disc", state=ModelState.DISCOVERED))
    removed = reg.prune(dry_run=True)
    assert len(removed) == 1
    assert len(reg) == 1
