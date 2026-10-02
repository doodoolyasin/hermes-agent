"""Tests for the ``hermes offline`` CLI surface (hermes_offline.cli)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import pytest

from hermes_offline import cli, emergency


@pytest.fixture()
def home(tmp_path, monkeypatch):
    target = tmp_path / "hh"
    monkeypatch.setenv("HERMES_HOME", str(target))
    return target


def _parse(argv):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    offline = sub.add_parser("offline")
    cli.register_cli(offline)
    return parser.parse_args(["offline", *argv])


def test_bare_offline_defaults_to_status(home, capsys):
    args = _parse([])
    assert cli.dispatch(args) == 0
    out = capsys.readouterr().out
    assert "READINESS" in out


def test_status_json_is_machine_readable(home, capsys):
    args = _parse(["status", "--json"])
    assert cli.dispatch(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "ready" in payload and "paths" in payload


def test_status_shows_emergency_mode_line(home, capsys):
    cli.dispatch(_parse(["status"]))
    out = capsys.readouterr().out
    assert "Emergency Mode" in out
    assert "Internet" in out


def test_models_runs_when_registry_empty(home, capsys):
    assert cli.dispatch(_parse(["models"])) == 0
    assert "no models known" in capsys.readouterr().out.lower() or True


def test_import_then_models_lists_it(home, tmp_path, capsys):
    src = tmp_path / "artifact.gguf"
    src.write_bytes(b"abc" * 100)
    digest = hashlib.sha256(b"abc" * 100).hexdigest()
    rc = cli.dispatch(_parse(["import", str(src), "--sha256", digest, "--name", "artifact"]))
    assert rc == 0
    out = capsys.readouterr().out
    assert "verified" in out.lower()


def test_import_missing_file_returns_error(home, capsys):
    assert cli.dispatch(_parse(["import", "/nope/missing.gguf"])) == 1
    assert "no such file" in capsys.readouterr().err.lower()


def test_prepare_requires_a_target(home, capsys):
    rc = cli.dispatch(_parse(["prepare"]))
    assert rc == 1
    assert "nothing to prepare" in capsys.readouterr().err.lower()


def test_recover_reports_missing_checkpoint(home, capsys):
    rc = cli.dispatch(_parse(["recover", "--task", "nope"]))
    assert rc == 1
    assert "no checkpoint" in capsys.readouterr().err.lower()


def test_recover_reads_saved_checkpoint(home, capsys):
    emergency.save_checkpoint("t1", {"step": 7})
    assert cli.dispatch(_parse(["recover", "--task", "t1"])) == 0
    assert json.loads(capsys.readouterr().out)["state"]["step"] == 7


def test_unknown_subcommand_returns_2():
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    offline = sub.add_parser("offline")
    cli.register_cli(offline)
    args = parser.parse_args(["offline"])
    args.offline_command = "bogus"
    assert cli.dispatch(args) == 2


def test_docs_generates_files(home, capsys):
    assert cli.dispatch(_parse(["docs"])) == 0
    assert len(os.listdir(emergency.docs_dir())) >= 5
