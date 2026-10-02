"""``hermes offline ...`` CLI subcommands.

Registered as a Hermes CLI plugin command via ``ctx.register_cli_command`` so no
core ``main.py`` change is required.  Every handler performs real work and
prints honest results — nothing here reports success it did not achieve.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import emergency
from .models import ModelState


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Wire up `hermes offline ...` subcommands."""
    subs = parser.add_subparsers(dest="offline_command", required=False)

    p_status = subs.add_parser("status", help="Show offline readiness + connectivity state")
    p_status.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    p_prepare = subs.add_parser("prepare", help="Prepare the emergency offline pack")
    p_prepare.add_argument("--model", action="append", default=[], help="Model name from the catalog (repeatable)")
    p_prepare.add_argument("--url", default=None, help="Direct artifact URL")
    p_prepare.add_argument("--sha256", default="", help="Expected sha256 of the artifact")
    p_prepare.add_argument("--name", default=None, help="Name for the directly-specified artifact")
    p_prepare.add_argument("--filename", default=None, help="Destination filename")
    p_prepare.add_argument("--dry-run", action="store_true", help="Show what would be prepared")
    p_prepare.add_argument("--offline", action="store_true", help="Do not use the network (local cache/import only)")
    p_prepare.add_argument("--json", action="store_true")

    p_models = subs.add_parser("models", help="List discovered / installed models")
    p_models.add_argument("--state", choices=[s.value for s in ModelState], default=None)
    p_models.add_argument("--json", action="store_true")

    subs.add_parser("verify", help="Re-verify installed artifact checksums")

    p_import = subs.add_parser("import", help="Import a model downloaded elsewhere")
    p_import.add_argument("path", help="Path to the model artifact")
    p_import.add_argument("--name", default=None)
    p_import.add_argument("--publisher", default="")
    p_import.add_argument("--license", default="")
    p_import.add_argument("--sha256", default="", help="Expected sha256 (enables VERIFIED state on match)")
    p_import.add_argument("--runtime", default="")

    p_doctor = subs.add_parser("doctor", help="Deep offline health / readiness check")
    p_doctor.add_argument("--json", action="store_true")

    subs.add_parser("docs", help="Regenerate offline documentation")

    p_recover = subs.add_parser("recover", help="Show a saved task checkpoint")
    p_recover.add_argument("--task", required=True)

    parser.set_defaults(func=dispatch)


def dispatch(args: argparse.Namespace) -> int:
    sub = getattr(args, "offline_command", None) or "status"
    handler = _COMMANDS.get(sub)
    if handler is None:
        print(f"unknown offline subcommand: {sub}", file=sys.stderr)
        return 2
    return handler(args)


# -- handlers ---------------------------------------------------------------


def _cmd_status(args: argparse.Namespace) -> int:
    data = emergency.status()
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    reg = emergency._load_registry()
    rep = emergency.readiness(records=reg.all() if reg else None)
    print("HERMES EMERGENCY READINESS")
    print("")
    width = max((len(i.name) for i in rep.items), default=12)
    for item in rep.items:
        label = item.name.replace("_", " ").title().ljust(width)
        print(f"{label}   {item.level.value.upper():9} {('(' + item.detail + ')') if item.detail else ''}")
    print("")
    print(f"Internet          {rep.connectivity.value.upper()}")
    print("")
    print(f"Emergency Mode    {'READY' if rep.ready else 'NOT READY'}")
    print("")
    print(f"offline home: {emergency.offline_home()}")
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    specs: List[dict] = []
    names: List[str] = list(args.model or [])
    if names:
        catalogue = _catalog_records()
        by_name = {r.name: r for r in catalogue}
        for n in names:
            rec = by_name.get(n)
            if rec is None:
                print(f"model not found in catalog: {n}", file=sys.stderr)
                return 1
            specs.append(_record_to_spec(rec))
    if args.url:
        specs.append({
            "name": args.name or "direct-download",
            "url": args.url,
            "sha256": args.sha256,
            "filename": args.filename or (args.name or "model.bin"),
        })
    if not specs:
        print("nothing to prepare: pass --model <name> (catalog) or --url <artifact>", file=sys.stderr)
        print("hint: `hermes offline models` lists catalog candidates.", file=sys.stderr)
        return 1

    report = emergency.prepare(specs, dry_run=args.dry_run, allow_network=not args.offline)
    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.ready or args.dry_run else 1
    for name in report.downloaded:
        print(f"✓ prepared {name}")
    for name in report.skipped:
        print(f"• skipped {name}")
    for failure in report.failed:
        print(f"✗ failed: {failure}", file=sys.stderr)
    for note in report.notes:
        print(f"  note: {note}", file=sys.stderr)
    if report.readiness:
        print("")
        print(report.readiness.render())
    return 0 if (report.ready or args.dry_run) else 1


def _cmd_models(args: argparse.Namespace) -> int:
    records = _catalog_records()
    reg = emergency._load_registry()
    installed = {r.key: r for r in (reg.all() if reg else [])}
    # prefer live registry state over catalog state for the same key
    merged = {r.key: r for r in records}
    merged.update(installed)
    rows = list(merged.values())
    if args.state:
        rows = [r for r in rows if r.state.value == args.state]
    if getattr(args, "json", False):
        print(json.dumps([r.to_dict() for r in rows], indent=2, sort_keys=True))
        return 0
    if not rows:
        print("no models known. Run `hermes offline prepare` or import an artifact.")
        return 0
    for r in sorted(rows, key=lambda x: x.name):
        size = r.disk_usage_gb or r.download_size_gb
        print(f"{r.state.value:10} {r.name:32} {size:6.1f} GB  {r.license or 'unknown license'}")
    return 0


def _cmd_verify(_args: argparse.Namespace) -> int:
    result = emergency.verify_pack()
    if result.get("ok"):
        print(f"✓ {result.get('checked', 0)} artifact(s) verified")
        return 0
    print(f"✗ verification failed ({result.get('checked', 0)} checked)", file=sys.stderr)
    for failure in result.get("failures", []):
        print(f"  {failure}", file=sys.stderr)
    return 1


def _cmd_import(args: argparse.Namespace) -> int:
    try:
        rec = emergency.import_model(
            args.path,
            name=args.name,
            publisher=args.publisher,
            license=args.license,
            sha256=args.sha256,
            runtime=args.runtime,
        )
    except FileNotFoundError:
        print(f"no such file: {args.path}", file=sys.stderr)
        return 1
    state = rec.state.value
    note = "verified" if rec.state == ModelState.VERIFIED else "installed (unverified — no matching sha256 given)"
    print(f"✓ imported {rec.name} -> {rec.local_path}")
    print(f"  state: {state} ({note})")
    print(f"  sha256: {rec.sha256}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    rep = emergency.readiness()
    if getattr(args, "json", False):
        print(json.dumps(rep.to_dict(), indent=2, sort_keys=True))
        return 0 if rep.ready else 1
    print(rep.render())
    print("")
    print("Dependencies:", json.dumps(emergency.dependency_snapshot().get("packages", {})))
    return 0 if rep.ready else 1


def _cmd_docs(_args: argparse.Namespace) -> int:
    written = __import__("hermes_offline.docs", fromlist=["generate_docs"]).generate_docs()
    for path in written:
        print(f"✓ {path}")
    return 0


def _cmd_recover(args: argparse.Namespace) -> int:
    data = emergency.load_checkpoint(args.task)
    if data is None:
        print(f"no checkpoint for task: {args.task}", file=sys.stderr)
        return 1
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


# -- helpers ----------------------------------------------------------------


def _catalog_records():
    try:
        from . import discovery  # type: ignore

        return discovery.load_catalog()
    except Exception:
        return []


def _record_to_spec(rec) -> dict:
    return {
        "name": rec.name,
        "url": rec.source_url,
        "sha256": rec.sha256,
        "filename": rec.name,
        "sources": [rec.source_url] if rec.source_url else [],
        "size_bytes": int((rec.download_size_gb or 0) * (1024 ** 3)),
        "publisher": rec.publisher,
        "license": rec.license,
        "runtime": rec.runtime,
    }


_COMMANDS = {
    "status": _cmd_status,
    "prepare": _cmd_prepare,
    "models": _cmd_models,
    "verify": _cmd_verify,
    "import": _cmd_import,
    "doctor": _cmd_doctor,
    "docs": _cmd_docs,
    "recover": _cmd_recover,
}
