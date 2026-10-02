"""Integration wiring: `hermes offline` and `hermes doctor --offline`."""

from __future__ import annotations

import argparse

from hermes_cli.subcommands.doctor import build_doctor_parser
from hermes_cli.subcommands.offline import build_offline_parser


def _subparsers():
    parser = argparse.ArgumentParser(prog="hermes")
    return parser, parser.add_subparsers(dest="command")


def test_offline_command_is_registered_as_builtin():
    from hermes_cli.main import _BUILTIN_SUBCOMMANDS

    assert "offline" in _BUILTIN_SUBCOMMANDS


def test_build_offline_parser_creates_dispatchable_subcommand():
    parser, sub = _subparsers()
    build_offline_parser(sub)
    assert "offline" in sub.choices
    args = parser.parse_args(["offline", "status"])
    assert args.offline_command == "status"
    assert callable(args.func)


def test_doctor_offline_flag_routes_to_offline_doctor(monkeypatch, capsys):
    called = {"standard": False}

    def fake_cmd(args):  # noqa: ANN001
        called["standard"] = True
        print("STANDARD-DOCTOR")
        return 0

    parser, sub = _subparsers()
    build_doctor_parser(sub, cmd_doctor=fake_cmd)

    # plain doctor still routes to the standard implementation
    args_std = parser.parse_args(["doctor"])
    args_std.func(args_std)
    assert called["standard"] is True
    capsys.readouterr()  # discard the standard doctor's output before the next run

    # --offline must NOT fall through to the standard doctor
    called["standard"] = False
    args_off = parser.parse_args(["doctor", "--offline"])
    rc = args_off.func(args_off)
    assert called["standard"] is False
    assert rc in (0, 1)
    assert "STANDARD-DOCTOR" not in capsys.readouterr().out


def test_real_cli_parser_exposes_offline_and_doctor_flag():
    """Build the *real* ``hermes`` parser: both integrations must be live."""
    from hermes_cli.main import _build_cli_parser

    parser, subparsers = _build_cli_parser()

    # `hermes offline ...` is a first-class builtin command.
    assert "offline" in subparsers.choices
    args = parser.parse_args(["offline", "status"])
    assert args.offline_command == "status"
    assert callable(args.func)

    # `hermes doctor --offline` parses.
    doc = parser.parse_args(["doctor", "--offline"])
    assert doc.offline is True
