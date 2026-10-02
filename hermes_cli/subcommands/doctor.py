"""``hermes doctor`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def _wrap_doctor(cmd_doctor: Callable) -> Callable:
    """Return the doctor dispatcher.

    With ``--offline`` the run reports the offline / emergency-mode readiness
    section (local model, runtime, recovery, dependency pack, connectivity)
    instead of the standard configuration diagnosis.  The hook lives here so
    the core doctor implementation is untouched.
    """

    def _dispatch(args):  # noqa: ANN001
        if getattr(args, "offline", False):
            try:
                from hermes_offline.cli import _cmd_doctor as _offline_doctor

                return _offline_doctor(args)
            except Exception as exc:  # noqa: BLE001
                print(f"offline doctor unavailable: {exc}")
                return 1
        return cmd_doctor(args)

    return _dispatch


def build_doctor_parser(subparsers, *, cmd_doctor: Callable) -> None:
    """Attach the ``doctor`` subcommand to ``subparsers``."""
    doctor_parser = subparsers.add_parser(
        "doctor", help="Check configuration and dependencies",
        description="Diagnose issues with Hermes Agent setup")
    doctor_parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix issues automatically")
    doctor_parser.add_argument(
        "--live", action="store_true",
        help="Opt-in: run one bounded, read-only real-call health probe per "
            "configured tool backend (Firecrawl/FAL/browser/MCP/TTS/STT) "
            "after the static checks. Makes real network calls.")
    doctor_parser.add_argument(
        "--offline", action="store_true",
        help="Report offline / emergency-mode readiness (local model, runtime, "
            "recovery, dependency pack, connectivity) instead of the standard "
            "configuration diagnosis.")
    doctor_parser.add_argument(
        "--ack", metavar="ADVISORY_ID", default=None,
        help="Acknowledge a security advisory by ID and exit. After ack, the "
            "advisory will no longer trigger startup banners. Run `hermes "
            "doctor` first to see active advisories and their IDs.")
    doctor_parser.set_defaults(func=_wrap_doctor(cmd_doctor))
