"""Tests for hermes_offline.docs (offline documentation generation)."""

from __future__ import annotations

import os

import pytest

from hermes_offline import docs, emergency


@pytest.fixture()
def home(tmp_path, monkeypatch):
    target = tmp_path / "hh"
    monkeypatch.setenv("HERMES_HOME", str(target))
    return target


def test_generate_docs_writes_core_files(home):
    written = docs.generate_docs([])
    names = {os.path.basename(p) for p in written}
    assert {"README.md", "models.md", "runtimes.md", "limitations.md"} <= names
    for path in written:
        assert os.path.getsize(path) > 0


def test_docs_do_not_claim_unimplemented_features(home):
    docs.generate_docs([])
    readme = open(os.path.join(emergency.docs_dir(), "README.md"), encoding="utf-8").read()
    # documents real commands
    assert "hermes offline prepare" in readme
    assert "EMERGENCY_OFFLINE_READY" in readme


def test_models_md_reflects_empty_registry(home):
    docs.generate_docs([])
    body = open(os.path.join(emergency.docs_dir(), "models.md"), encoding="utf-8").read()
    assert "No models are registered yet" in body


def test_docs_regenerated_overwrite(home):
    docs.generate_docs([])
    first = os.path.getsize(os.path.join(emergency.docs_dir(), "models.md"))
    docs.generate_docs([])
    assert os.path.getsize(os.path.join(emergency.docs_dir(), "models.md")) == first
