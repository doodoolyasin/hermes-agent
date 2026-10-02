"""Tests for hermes_offline.discovery (offline catalog + discovery)."""

from __future__ import annotations

import json
import os

import pytest

from hermes_offline import discovery, registry
from hermes_offline.models import Channel, HardwareProfile, ModelRecord, ModelState


@pytest.fixture()
def home(tmp_path, monkeypatch):
    target = tmp_path / "hh"
    monkeypatch.setenv("HERMES_HOME", str(target))
    return target


# -- bundled catalog -------------------------------------------------------


def test_bundled_catalog_has_valid_unique_records():
    records = discovery.bundled_catalog()
    assert len(records) >= 5
    keys = [r.key for r in records]
    assert len(keys) == len(set(keys))  # unique identity
    for r in records:
        assert r.name
        assert r.publisher
        assert r.license
        assert r.format == "gguf"
        assert r.runtime == "llama.cpp"
        assert r.capabilities
        assert r.requirements.download_size_gb > 0
        assert r.extra.get("catalog") == "bundled"


def test_bundled_catalog_never_invents_checksums():
    """Honesty invariant: we only trust operator-supplied hashes."""
    assert all(r.sha256 == "" for r in discovery.bundled_catalog())


def test_bundled_catalog_is_fresh_each_call():
    first = discovery.bundled_catalog()
    first[0].capabilities.append("mutated")
    first[0].extra["x"] = 1
    second = discovery.bundled_catalog()
    assert "mutated" not in second[0].capabilities
    assert "x" not in second[0].extra


# -- loading / merging -----------------------------------------------------


def test_load_catalog_includes_bundled_by_default(home):
    records = discovery.load_catalog()
    assert {r.key for r in records} == {r.key for r in discovery.bundled_catalog()}


def test_load_catalog_file_is_tolerant(tmp_path):
    assert discovery.load_catalog_file(str(tmp_path / "missing.json")) == []
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert discovery.load_catalog_file(str(bad)) == []


def test_user_catalog_overrides_bundled_and_adds(home):
    bundled = discovery.bundled_catalog()[0]
    override = ModelRecord.from_dict(bundled.to_dict())
    override.license = "Custom-1.0"
    extra = ModelRecord(name="My-Private-Model", publisher="me", license="Proprietary", format="gguf")
    path = discovery.save_catalog([override, extra])

    records = discovery.load_catalog()
    by_key = {r.key: r for r in records}
    assert by_key[bundled.key].license == "Custom-1.0"
    assert by_key[extra.key].license == "Proprietary"
    # overriding must not duplicate the key
    assert sum(1 for r in records if r.key == bundled.key) == 1


def test_merge_catalogs_later_wins():
    a = ModelRecord(name="m", publisher="p", license="A")
    b = ModelRecord(name="m", publisher="p", license="B")
    merged = discovery.merge_catalogs([a], [b])
    assert len(merged) == 1
    assert merged[0].license == "B"


def test_save_catalog_is_atomic(tmp_path):
    path = discovery.save_catalog([ModelRecord(name="m", publisher="p")], path=str(tmp_path / "c.json"))
    assert os.path.exists(path)
    assert [f for f in os.listdir(tmp_path) if ".tmp" in f] == []
    assert discovery.load_catalog_file(path)[0].name == "m"


def test_extra_paths_and_env_are_honoured(home, tmp_path, monkeypatch):
    env_catalog = tmp_path / "env.json"
    env_catalog.write_text(json.dumps({"models": [{"name": "env-model", "publisher": "p"}]}), encoding="utf-8")
    monkeypatch.setenv("HERMES_OFFLINE_CATALOG", str(env_catalog))
    names = {r.name for r in discovery.load_catalog()}
    assert "env-model" in names


# -- discover / recommend --------------------------------------------------


def test_discover_filters_by_capability():
    coding = discovery.discover(capabilities=["coding"])
    assert coding, "expected at least one coding model"
    assert all("coding" in r.capabilities for r in coding)
    # every returned record is from the full catalog
    assert all(r.key in {b.key for b in discovery.bundled_catalog()} for r in coding)


def test_discover_filters_by_runtime_and_size():
    small = discovery.discover(max_download_gb=2.0)
    assert small
    assert all(r.download_size_gb <= 2.0 for r in small)


def test_discover_require_fit_excludes_oversized_models():
    tiny = HardwareProfile(ram_total_gb=4.0, ram_available_gb=3.0, disk_free_gb=5.0)
    fitted = discovery.discover(hardware=tiny, require_fit=True)
    assert fitted
    assert all(r.requirements.ram_gb * 1.25 <= 4.0 for r in fitted)
    assert all(r.download_size_gb <= 5.0 for r in fitted)
    # the full catalog has models too big for this machine
    assert len(fitted) < len(discovery.bundled_catalog())


def test_discover_annotates_fit_without_require(tmp_path):
    hw = HardwareProfile(ram_total_gb=4.0, ram_available_gb=3.0, disk_free_gb=100.0)
    records = discovery.discover(hardware=hw)
    assert any(r.extra.get("fit") is False for r in records)
    assert any(r.extra.get("fit") is True for r in records)


def test_discover_merges_registry_state_over_catalog(tmp_path):
    bundled = discovery.bundled_catalog()[0]
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    installed = ModelRecord.from_dict(bundled.to_dict())
    installed.state = ModelState.VERIFIED
    installed.local_path = "/models/installed.gguf"
    installed.sha256 = "c" * 64
    reg.add_discovered(installed, replace=True)

    records = discovery.discover(registry=reg)
    got = next(r for r in records if r.key == bundled.key)
    assert got.state == ModelState.VERIFIED
    assert got.local_path == "/models/installed.gguf"


def test_discover_state_filter(tmp_path):
    reg = registry.ModelRegistry(str(tmp_path / "r.json"))
    reg.add_discovered(ModelRecord(name="v", publisher="p", state=ModelState.VERIFIED, local_path="/v"))
    records = discovery.discover(registry=reg, states=[ModelState.VERIFIED], include_registry=True)
    assert [r.name for r in records] == ["v"]


def test_recommend_excludes_models_that_do_not_fit():
    hw = HardwareProfile(ram_total_gb=4.0, ram_available_gb=3.5, disk_free_gb=10.0)
    picks = discovery.recommend(hw, limit=5)
    assert picks
    assert all(r.requirements.download_size_gb <= 10.0 for r in picks)
    assert all(r.requirements.ram_gb * 1.25 <= 4.0 for r in picks)


def test_recommend_ranks_capability_coverage_first():
    hw = HardwareProfile(ram_total_gb=32.0, ram_available_gb=28.0, disk_free_gb=100.0, has_gpu=True, vram_total_gb=16.0)
    picks = discovery.recommend(hw, capabilities=["coding", "tools"], limit=3)
    assert picks
    # The top pick covers both requested capabilities.
    top_caps = {c.lower() for c in picks[0].capabilities}
    assert {"coding", "tools"} <= top_caps


# -- serialisation ---------------------------------------------------------


def test_catalog_data_roundtrip():
    records = discovery.bundled_catalog()[:3]
    data = discovery.catalog_to_data(records)
    parsed = discovery.catalog_from_data(data)
    assert [r.key for r in parsed] == [r.key for r in records]
    assert parsed[0].requirements.ram_gb == records[0].requirements.ram_gb
    assert parsed[0].channel in (Channel.STABLE, Channel.EXPERIMENTAL)


def test_catalog_from_data_skips_bad_rows():
    parsed = discovery.catalog_from_data({"models": [{"name": "ok", "publisher": "p"}, {"no_name": True}, "junk"]})
    assert [r.name for r in parsed] == ["ok"]
