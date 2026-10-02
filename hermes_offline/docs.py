"""Offline documentation generator.

Everything written here must be readable with no internet connection, so the
files are plain Markdown generated from *actual* local state (registry,
detected runtimes, readiness).  We never document a feature that is not
implemented, and never invent model facts.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from . import emergency
from .models import ModelState

_README = """# Hermes Offline Mode

This directory is generated automatically by `hermes offline prepare` and is
intended to be readable **without any network connection**.

## Commands

| Command | Purpose |
| --- | --- |
| `hermes offline status` | Show readiness + connectivity state |
| `hermes offline prepare` | Download and prepare the emergency pack |
| `hermes offline models` | List known / installed models |
| `hermes offline verify` | Re-verify artifact checksums |
| `hermes offline import <path>` | Import a model downloaded elsewhere |
| `hermes offline doctor` | Deep offline health check |
| `hermes offline docs` | Regenerate this documentation |

## How offline mode works

```
Cloud provider  --fail-->  Secondary provider  --fail-->  Local provider
                                                             |
                                                     Emergency Offline Mode
```

When all network connectivity is unavailable, Hermes can fall back to a locally
prepared model served by a local runtime (Ollama or llama.cpp).  Nothing in the
offline path depends on DNS, cloud APIs, licence checks or telemetry.

## Connectivity states

* **FULL_ONLINE** — everything reachable and fast
* **DEGRADED** — some endpoints slow or intermittently failing
* **PROVIDER_UNAVAILABLE** — network works, the model provider does not
* **OFFLINE** — no network at all
* **EMERGENCY_OFFLINE_READY** — offline *and* a local fallback is prepared

See `limitations.md` for what offline mode deliberately does not do.
"""

_LIMITATIONS = """# Offline Limitations

Offline mode is honest about what it cannot do:

* Local models are smaller than frontier cloud models; expect lower quality.
* Tool calls, vision and long context depend on the specific local model.
* Anything that genuinely needs the network (web search, remote APIs, model
  downloads) reports **NETWORK REQUIRED** instead of hanging.
* The emergency pack only contains what you explicitly prepared with
  `hermes offline prepare`.
"""


def _models_md(records: List[Any]) -> str:
    lines = ["# Local Models", ""]
    if not records:
        lines += ["No models are registered yet.", "", "Run `hermes offline prepare` first."]
        return "\n".join(lines)
    for rec in sorted(records, key=lambda r: r.name):
        lines.append(f"## {rec.name}")
        lines.append("")
        lines.append(f"- publisher: {rec.publisher or 'unknown'}")
        lines.append(f"- state: {rec.state.value}")
        lines.append(f"- license: {rec.license or 'unknown'}")
        lines.append(f"- runtime: {rec.runtime or 'unset'}")
        lines.append(f"- quantization: {rec.quantization or 'n/a'}")
        lines.append(f"- size: {rec.download_size_gb or rec.disk_usage_gb:.1f} GB")
        lines.append(f"- context: {rec.context_length or 'unknown'}")
        lines.append(f"- path: {rec.local_path or '(not installed)'}")
        lines.append(f"- sha256: {rec.sha256 or '(unknown)'}")
        if rec.capabilities:
            lines.append(f"- capabilities: {', '.join(rec.capabilities)}")
        lines.append("")
    return "\n".join(lines)


def _runtimes_md() -> str:
    lines = ["# Local Runtimes", ""]
    try:
        from .runtimes import RuntimeManager

        infos = RuntimeManager.from_config({}).detect()
    except Exception as exc:
        return "# Local Runtimes\n\nRuntime subsystem unavailable: " + str(exc)
    if not infos:
        lines.append("No runtimes detected.")
    for info in infos:
        lines.append(f"## {info.name}")
        lines.append("")
        lines.append(f"- available: {'yes' if info.available else 'no'}")
        lines.append(f"- endpoint: {info.endpoint or 'n/a'}")
        lines.append(f"- models: {', '.join(info.models) if info.models else '(none listed)'}")
        lines.append(f"- detail: {info.detail}")
        lines.append("")
    return "\n".join(lines)


def _recovery_md() -> str:
    return "\n".join([
        "# Recovery",
        "",
        "Interrupted tasks are checkpointed under:",
        "",
        f"    {emergency.checkpoints_dir()}",
        "",
        "Checkpoints are written atomically (temp file + rename). After a crash,",
        "process restart or provider failure, a task can resume from its last",
        "checkpoint instead of starting over.",
    ])


def _config_md() -> str:
    return "\n".join([
        "# Configuration",
        "",
        "All connectivity knobs are configurable — no proxy, host or mirror is",
        "hardcoded. Relevant environment variables:",
        "",
        "| Variable | Purpose |",
        "| --- | --- |",
        "| `HERMES_HOME` | Hermes state directory (default `~/.hermes`) |",
        "| `HERMES_OFFLINE_PROXY` | Optional HTTP/SOCKS proxy for downloads |",
        "| `HERMES_OFFLINE_MIRRORS` | Comma-separated mirror base URLs |",
        "| `BALE_BOT_TOKEN` | Bale messenger bot token (gateway) |",
        "| `BALE_API_BASE` | Bale Bot API base (default `https://tapi.bale.ai`) |",
        "",
        "Never commit these values to source control.",
    ])


def generate_docs(records: Optional[List[Any]] = None) -> List[str]:
    """Write the offline documentation set; returns the file paths written."""
    out = emergency.docs_dir()
    os.makedirs(out, exist_ok=True)
    if records is None:
        reg = emergency._load_registry()
        records = reg.all() if reg is not None else []

    docs: Dict[str, str] = {
        "README.md": _README,
        "models.md": _models_md(records),
        "runtimes.md": _runtimes_md(),
        "recovery.md": _recovery_md(),
        "configuration.md": _config_md(),
        "limitations.md": _LIMITATIONS,
    }
    written: List[str] = []
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    for fname, body in docs.items():
        path = os.path.join(out, fname)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body.rstrip() + f"\n\n<!-- generated {stamp} -->\n")
        written.append(path)
    return written
