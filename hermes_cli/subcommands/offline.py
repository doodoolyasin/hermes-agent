"""``hermes offline`` subcommand parser.

First-class CLI surface for offline / emergency operation.  The implementation
lives in :mod:`hermes_offline.cli`; this module only attaches it to the Hermes
argparse tree using the same convention as the other ``hermes_cli.subcommands``
builders.
"""

from __future__ import annotations


def build_offline_parser(subparsers) -> None:
    """Attach the ``offline`` subcommand to ``subparsers``."""
    offline_parser = subparsers.add_parser(
        "offline",
        help="Offline / emergency mode: readiness, model pack, local fallback",
        description=(
            "Prepare and operate Hermes without international (or any) "
            "connectivity. Manages the emergency offline pack (local model + "
            "runtime), classified connectivity states, artifact verification, "
            "crash recovery and offline documentation."
        ),
    )

    from hermes_offline.cli import register_cli

    register_cli(offline_parser)
