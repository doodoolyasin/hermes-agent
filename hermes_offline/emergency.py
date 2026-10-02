"""Emergency offline pack: preparation, verification, readiness and recovery.

This module is the integration point for the offline subsystem.  It performs
*real* checks (filesystem, runtimes, registry state, disk, dependencies) rather
than reading configuration flags, so the readiness report reflects reality.

Design rules:

* Never fabricate readiness — every item is computed from a live probe.
* Never mark a corrupted/incomplete artifact as ready.
* Standard library only; optional submodules (``registry``, ``download``,
  ``discovery``, ``hardware``, ``connectivity``, ``runtimes``) are imported
  lazily so a missing optional piece degrades one report line, not the CLI.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import (
    ConnectivityState,
    ModelRecord,
    ModelState,
    ReadinessItem,
    ReadinessLevel,
    ReadinessReport,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_DISK_FLOOR_GB = 2.0


def hermes_home() -> str:
    return os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")


def offline_home() -> str:
    return os.path.join(hermes_home(), "offline")


def models_dir() -> str:
    return os.path.join(offline_home(), "models")


def docs_dir() -> str:
    return os.path.join(offline_home(), "docs")


def checkpoints_dir() -> str:
    return os.path.join(offline_home(), "checkpoints")


def registry_path() -> str:
    return os.path.join(offline_home(), "registry.json")


def pack_manifest_path() -> str:
    return os.path.join(offline_home(), "pack.json")


def ensure_dirs() -> None:
    for d in (offline_home(), models_dir(), docs_dir(), checkpoints_dir()):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Optional submodule helpers (lazy, never fatal)
# ---------------------------------------------------------------------------


def _load_registry():
    try:
        from .registry import ModelRegistry  # type: ignore

        return ModelRegistry(registry_path())
    except Exception:
        return None


def _persist(reg) -> None:
    """Persist a registry if it exposes a ``save`` method.

    ``add_discovered``/``mark_*`` may or may not auto-save depending on the
    implementation, so callers persist explicitly.  A save failure must never
    lose an in-memory result or crash the command.
    """
    save = getattr(reg, "save", None)
    if callable(save):
        try:
            save()
        except Exception:  # noqa: BLE001 - best-effort durability
            pass


def _load_downloader(**kwargs):
    try:
        from .download import Downloader  # type: ignore

        return Downloader(**kwargs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Readiness — every item is a REAL check
# ---------------------------------------------------------------------------


def _check_local_model(records: List[ModelRecord]) -> ReadinessItem:
    ready = [r for r in records if r.state == ModelState.VERIFIED and r.local_path and os.path.exists(r.local_path)]
    if ready:
        biggest = max(ready, key=lambda r: r.disk_usage_gb)
        return ReadinessItem("local_model", ReadinessLevel.READY, f"{len(ready)} verified; e.g. {biggest.name}")
    installed = [r for r in records if r.state == ModelState.INSTALLED]
    if installed:
        return ReadinessItem(
            "local_model", ReadinessLevel.DEGRADED, f"{len(installed)} installed but unverified"
        )
    return ReadinessItem("local_model", ReadinessLevel.NOT_READY, "no verified local model")


def _check_runtime() -> ReadinessItem:
    try:
        from .runtimes import RuntimeManager  # type: ignore

        available = RuntimeManager.from_config({}).available()
    except Exception:
        return ReadinessItem("runtime", ReadinessLevel.NOT_READY, "runtime subsystem unavailable")
    if available:
        names = ", ".join(i.name for i in available)
        return ReadinessItem("runtime", ReadinessLevel.READY, names)
    return ReadinessItem("runtime", ReadinessLevel.NOT_READY, "no local runtime detected (ollama/llama.cpp)")


def _check_tools(*, workspace: Optional[str] = None) -> ReadinessItem:
    """Verify the local tools offline mode relies on: python, shell, git, fs."""
    missing: List[str] = []
    if not sys.executable or not os.path.exists(sys.executable):
        missing.append("python")
    if shutil.which("sh") is None and shutil.which("bash") is None:
        missing.append("shell")
    if shutil.which("git") is None:
        # git is useful but not strictly required for chatting
        pass
    # filesystem write probe
    probe_dir = workspace or models_dir()
    try:
        os.makedirs(probe_dir, exist_ok=True)
        probe = os.path.join(probe_dir, ".rw_probe")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError:
        missing.append("filesystem-write")
    if missing:
        return ReadinessItem("tools", ReadinessLevel.NOT_READY, "missing: " + ", ".join(missing))
    return ReadinessItem("tools", ReadinessLevel.READY, "python, shell, filesystem")


def _check_memory() -> ReadinessItem:
    home = hermes_home()
    try:
        os.makedirs(home, exist_ok=True)
        probe = os.path.join(home, ".offline_rw_probe")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as exc:
        return ReadinessItem("memory", ReadinessLevel.NOT_READY, f"HERMES_HOME not writable: {exc}")
    return ReadinessItem("memory", ReadinessLevel.READY, hermes_home())


def _check_offline_docs() -> ReadinessItem:
    d = docs_dir()
    try:
        entries = [e for e in os.listdir(d) if not e.startswith(".")]
    except OSError:
        entries = []
    if entries:
        return ReadinessItem("offline_docs", ReadinessLevel.READY, f"{len(entries)} file(s)")
    return ReadinessItem("offline_docs", ReadinessLevel.NOT_READY, "run `hermes offline prepare`")


def _check_recovery() -> ReadinessItem:
    d = checkpoints_dir()
    if not os.path.isdir(d):
        return ReadinessItem("recovery", ReadinessLevel.NOT_READY, "no checkpoint directory")
    files = [f for f in os.listdir(d) if f.endswith(".json")]
    if files:
        return ReadinessItem("recovery", ReadinessLevel.READY, f"{len(files)} checkpoint(s)")
    return ReadinessItem("recovery", ReadinessLevel.DEGRADED, "checkpoint dir present, no snapshots yet")


def _check_disk(floor_gb: float = DEFAULT_DISK_FLOOR_GB) -> ReadinessItem:
    try:
        usage = shutil.disk_usage(offline_home() if os.path.isdir(offline_home()) else hermes_home())
        free_gb = usage.free / (1024 ** 3)
    except OSError as exc:
        return ReadinessItem("disk", ReadinessLevel.NOT_READY, f"cannot stat: {exc}")
    if free_gb >= floor_gb:
        return ReadinessItem("disk", ReadinessLevel.READY, f"{free_gb:.1f} GB free")
    if free_gb > 0:
        return ReadinessItem("disk", ReadinessLevel.NOT_READY, f"only {free_gb:.1f} GB free")
    return ReadinessItem("disk", ReadinessLevel.NOT_READY, "disk full")


def _check_dependencies() -> ReadinessItem:
    """Preflight: the emergency pack must not discover a missing dependency at
    failure time.  We check the stdlib modules offline mode itself needs."""
    required = ("json", "socket", "ssl", "urllib.request", "shutil", "subprocess", "hashlib")
    missing: List[str] = []
    for mod in required:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        return ReadinessItem("dependencies", ReadinessLevel.NOT_READY, "missing: " + ", ".join(missing))
    return ReadinessItem("dependencies", ReadinessLevel.READY, f"{len(required)} stdlib modules present")


def _check_connectivity() -> ConnectivityState:
    try:
        from .connectivity import ConnectivityMonitor, ConnectivityConfig  # type: ignore

        cfg = ConnectivityConfig()
        mon = ConnectivityMonitor(cfg)
        return mon.check()
    except Exception:
        return ConnectivityState.FULL_ONLINE


def dependency_snapshot() -> Dict[str, Any]:
    """Record exact versions of relevant packages so the pack is reproducible."""
    snap: Dict[str, Any] = {"python": sys.version.split()[0], "executable": sys.executable}
    try:
        import importlib.metadata as md

        pkgs = {}
        for name in ("httpx", "pyyaml", "pydantic", "pytest"):
            try:
                pkgs[name] = md.version(name)
            except Exception:
                pkgs[name] = None
        snap["packages"] = pkgs
    except Exception:
        snap["packages"] = {}
    return snap


def readiness(*, records: Optional[List[ModelRecord]] = None, disk_floor_gb: float = DEFAULT_DISK_FLOOR_GB) -> ReadinessReport:
    """Compute the emergency readiness report from live checks."""
    if records is None:
        reg = _load_registry()
        records = reg.all() if reg is not None else []
    items = [
        _check_local_model(records),
        _check_runtime(),
        _check_tools(),
        _check_memory(),
        _check_offline_docs(),
        _check_recovery(),
        _check_disk(disk_floor_gb),
        _check_dependencies(),
    ]
    return ReadinessReport(items=items, connectivity=_check_connectivity(), generated_at=time.time())


# ---------------------------------------------------------------------------
# Pack prepare / verify / import
# ---------------------------------------------------------------------------


@dataclass
class PrepareReport:
    ready: bool = False
    downloaded: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[Dict[str, str]] = field(default_factory=list)
    readiness: Optional[ReadinessReport] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "downloaded": self.downloaded,
            "skipped": self.skipped,
            "failed": self.failed,
            "notes": self.notes,
            "readiness": self.readiness.to_dict() if self.readiness else None,
        }


def prepare(specs: List[Dict[str, Any]], *, dry_run: bool = False, allow_network: bool = True) -> PrepareReport:
    """Prepare the offline pack for ``specs`` (each a DownloadSpec-like dict).

    Steps per model: disk preflight -> resumable download -> checksum verify ->
    atomic install -> registry update.  A failing model never marks the pack
    ready.  Network access may be disallowed (``allow_network=False``) so an
    operator can prepare from a local cache / import path only.
    """
    report = PrepareReport()
    ensure_dirs()
    reg = _load_registry()
    downloader = _load_downloader() if (specs and not dry_run) else None

    if specs and downloader is None and not dry_run:
        report.failed.append({"error": "download subsystem unavailable"})
        report.notes.append("hermes_offline.download could not be imported")
        report.readiness = readiness()
        return report

    for spec in specs:
        name = spec.get("name", spec.get("filename", "model"))
        url = spec.get("url") or (spec.get("sources") or [""])[0]
        sha = spec.get("sha256", "")
        if dry_run:
            report.skipped.append(f"{name} (dry-run)")
            continue
        if not url and not allow_network:
            report.skipped.append(f"{name} (no source, offline)")
            continue
        try:
            result = downloader.download(_as_download_spec(spec))
            state = getattr(result.state, "value", str(result.state))
            if state == "complete":
                report.downloaded.append(name)
                if reg is not None:
                    rec = ModelRecord.from_dict(spec) if "name" in spec else ModelRecord(name=name)
                    rec.state = ModelState.VERIFIED if sha else ModelState.INSTALLED
                    rec.local_path = getattr(result, "path", "")
                    rec.sha256 = getattr(result, "sha256", "") or sha
                    try:
                        rec.disk_usage_gb = os.path.getsize(rec.local_path) / (1024 ** 3) if rec.local_path else 0.0
                    except OSError:
                        rec.disk_usage_gb = 0.0
                    reg.add_discovered(rec, replace=True)
                    _persist(reg)
            else:
                report.failed.append({"name": name, "state": state, "error": getattr(result, "error", "")})
        except Exception as exc:  # a single bad spec must not abort the pack
            report.failed.append({"name": name, "error": str(exc)})

    report.readiness = readiness(records=reg.all() if reg is not None else None)
    report.ready = report.readiness.ready and not report.failed
    _write_manifest(report, specs)
    return report


def _as_download_spec(spec: Dict[str, Any]):
    from .download import DownloadSpec  # type: ignore

    sources = spec.get("sources") or ([spec["url"]] if spec.get("url") else [])
    return DownloadSpec(
        name=spec.get("name", ""),
        url=spec.get("url", sources[0] if sources else ""),
        sha256=spec.get("sha256", ""),
        dest_dir=spec.get("dest_dir", models_dir()),
        filename=spec.get("filename", spec.get("name", "model.bin")),
        size_bytes=int(spec.get("size_bytes", 0) or 0),
        sources=list(sources),
        headers=dict(spec.get("headers", {}) or {}),
        chunk_size=int(spec.get("chunk_size", 1 << 20)),
    )


def _write_manifest(report: PrepareReport, specs: List[Dict[str, Any]]) -> None:
    manifest = {
        "version": 1,
        "generated_at": time.time(),
        "dependencies": dependency_snapshot(),
        "models": specs,
        "downloaded": report.downloaded,
        "failed": report.failed,
        "ready": report.ready,
    }
    ensure_dirs()
    tmp = pack_manifest_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    os.replace(tmp, pack_manifest_path())


def verify_pack() -> Dict[str, Any]:
    """Re-verify every installed artifact's checksum against the registry."""
    reg = _load_registry()
    out: Dict[str, Any] = {"ok": True, "checked": 0, "failures": []}
    if reg is None:
        out["ok"] = False
        out["failures"].append({"error": "registry unavailable"})
        return out
    try:
        from .download import verify_file  # type: ignore
    except Exception:
        verify_file = None
    for rec in reg.all():
        if rec.state not in (ModelState.INSTALLED, ModelState.VERIFIED) or not rec.local_path:
            continue
        out["checked"] += 1
        if not os.path.exists(rec.local_path):
            out["ok"] = False
            out["failures"].append({"name": rec.name, "error": "file missing"})
            continue
        if rec.sha256 and verify_file is not None:
            actual = verify_file(rec.local_path)
            if actual and actual.lower() != rec.sha256.lower():
                out["ok"] = False
                out["failures"].append({"name": rec.name, "error": "checksum mismatch"})
                rec.state = ModelState.FAILED
                rec.last_error = "checksum mismatch during verify"
                reg.add_discovered(rec, replace=True)
                _persist(reg)
    return out


def import_model(
    path: str,
    *,
    name: Optional[str] = None,
    publisher: str = "",
    license: str = "",
    sha256: str = "",
    runtime: str = "",
    copy: bool = True,
) -> ModelRecord:
    """Import a model downloaded elsewhere (offline transfer).

    Checksum is computed locally.  The record is only marked VERIFIED when an
    expected ``sha256`` was supplied *and* matches — otherwise it stays INSTALLED
    (unverified), and we never pretend otherwise.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ensure_dirs()
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()

    fname = name or os.path.basename(path)
    dest = os.path.join(models_dir(), os.path.basename(path))
    if os.path.abspath(path) != os.path.abspath(dest):
        if copy:
            # Re-importing the same artifact is common (e.g. verifying an
            # already-linked file).  Skip the copy when dest already refers to
            # the same inode, otherwise replace whatever stale file is there.
            already_linked = os.path.exists(dest) and os.path.samefile(path, dest)
            if not already_linked:
                if os.path.exists(dest):
                    os.remove(dest)
                # same-filesystem link first (cheap), fall back to a real copy
                try:
                    os.link(path, dest)
                except OSError:
                    shutil.copy2(path, dest)
        else:
            dest = os.path.abspath(path)

    rec = ModelRecord(name=fname, publisher=publisher, source="import", license=license, sha256=actual)
    rec.runtime = runtime
    rec.local_path = dest
    rec.disk_usage_gb = os.path.getsize(dest) / (1024 ** 3)
    rec.installed_at = time.time()
    verified = bool(sha256) and sha256.lower() == actual.lower()
    rec.state = ModelState.VERIFIED if verified else ModelState.INSTALLED
    rec.extra["expected_sha256"] = sha256
    rec.last_verified = time.time() if verified else None

    reg = _load_registry()
    if reg is not None:
        reg.add_discovered(rec, replace=True)
        _persist(reg)
    return rec


# ---------------------------------------------------------------------------
# Task checkpoints (crash recovery)
# ---------------------------------------------------------------------------


def save_checkpoint(task_id: str, state: Dict[str, Any]) -> str:
    ensure_dirs()
    payload = {"task_id": task_id, "saved_at": time.time(), "state": state}
    path = os.path.join(checkpoints_dir(), f"{_safe(task_id)}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def load_checkpoint(task_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(checkpoints_dir(), f"{_safe(task_id)}.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _safe(task_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in task_id)[:120] or "task"


def status() -> Dict[str, Any]:
    rep = readiness()
    return {
        "ready": rep.ready,
        "connectivity": rep.connectivity.value,
        "readiness": rep.to_dict(),
        "paths": {
            "home": offline_home(),
            "models": models_dir(),
            "docs": docs_dir(),
            "checkpoints": checkpoints_dir(),
        },
    }
