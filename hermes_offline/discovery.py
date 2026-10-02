"""Offline model discovery and metadata.

Discovery is **offline-first** by design: it reads a *bundled* catalog shipped
with Hermes plus an optional operator-supplied ``catalog.json`` under
``$HERMES_HOME/offline/``.  It never scrapes a model hub — a machine that is
already offline (or on a blocked link) must still be able to enumerate the
models it could use.

The bundled entries are curated, well-known open-weight models with honest
metadata.  Two deliberate choices:

* ``sha256`` is left empty.  We only trust checksums the operator supplies; an
  invented hash would be worse than none.
* Sizes are labelled as estimates (``extra['size_is_estimate']``) because a
  quantized artifact's exact byte count depends on the conversion.

The shared :class:`~hermes_offline.models.ModelRecord` contract is used for
every entry, so catalog entries flow through the same registry, downloader and
readiness code as imported artifacts.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .models import Channel, HardwareProfile, ModelRecord, ModelRequirements, ModelState

#: Schema version for catalog documents written/read by this module.
CATALOG_VERSION = 1
CATALOG_FILENAME = "catalog.json"


def offline_home() -> str:
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(home, "offline")


def catalog_path() -> str:
    """Operator-supplied catalog location (``$HERMES_HOME/offline/catalog.json``)."""
    return os.path.join(offline_home(), CATALOG_FILENAME)


# ---------------------------------------------------------------------------
# Bundled catalog
# ---------------------------------------------------------------------------
#
# Metadata only.  Runtime/download fields are derived below so the table stays
# readable and there is one place that knows how a table row maps to a record.
_BUNDLED: List[Dict[str, Any]] = [
    {
        "name": "Llama-3.2-3B-Instruct",
        "publisher": "meta-llama",
        "license": "Llama-3.2 Community License",
        "license_url": "https://www.llama.com/llama3_2/license/",
        "architecture": "llama",
        "parameter_count": "3B",
        "context_length": 131072,
        "quantization": "Q4_K_M",
        "ram_gb": 4.0,
        "vram_gb": 3.0,
        "download_size_gb": 2.0,
        "capabilities": ["chat", "tools", "multilingual"],
        "source_url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "publisher": "meta-llama",
        "license": "Llama-3.1 Community License",
        "license_url": "https://www.llama.com/llama3_1/license/",
        "architecture": "llama",
        "parameter_count": "8B",
        "context_length": 131072,
        "quantization": "Q4_K_M",
        "ram_gb": 9.0,
        "vram_gb": 7.0,
        "download_size_gb": 4.9,
        "capabilities": ["chat", "tools", "multilingual", "reasoning"],
        "source_url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    },
    {
        "name": "Qwen2.5-3B-Instruct",
        "publisher": "Qwen",
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "architecture": "qwen2",
        "parameter_count": "3B",
        "context_length": 32768,
        "quantization": "Q4_K_M",
        "ram_gb": 4.0,
        "vram_gb": 3.0,
        "download_size_gb": 2.0,
        "capabilities": ["chat", "tools", "multilingual", "coding"],
        "source_url": "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    },
    {
        "name": "Qwen2.5-7B-Instruct",
        "publisher": "Qwen",
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "architecture": "qwen2",
        "parameter_count": "7B",
        "context_length": 32768,
        "quantization": "Q4_K_M",
        "ram_gb": 9.0,
        "vram_gb": 7.0,
        "download_size_gb": 4.7,
        "capabilities": ["chat", "tools", "multilingual", "coding", "reasoning"],
        "source_url": "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    },
    {
        "name": "Qwen2.5-Coder-7B-Instruct",
        "publisher": "Qwen",
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "architecture": "qwen2",
        "parameter_count": "7B",
        "context_length": 32768,
        "quantization": "Q4_K_M",
        "ram_gb": 9.0,
        "vram_gb": 7.0,
        "download_size_gb": 4.7,
        "capabilities": ["coding", "tools", "chat"],
        "source_url": "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
    },
    {
        "name": "Mistral-7B-Instruct-v0.3",
        "publisher": "mistralai",
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "architecture": "mistral",
        "parameter_count": "7B",
        "context_length": 32768,
        "quantization": "Q4_K_M",
        "ram_gb": 9.0,
        "vram_gb": 7.0,
        "download_size_gb": 4.4,
        "capabilities": ["chat", "tools"],
        "source_url": "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    },
    {
        "name": "Phi-3.5-mini-instruct",
        "publisher": "microsoft",
        "license": "MIT",
        "license_url": "https://opensource.org/licenses/MIT",
        "architecture": "phi3",
        "parameter_count": "3.8B",
        "context_length": 131072,
        "quantization": "Q4_K_M",
        "ram_gb": 4.5,
        "vram_gb": 3.5,
        "download_size_gb": 2.4,
        "capabilities": ["chat", "coding", "reasoning"],
        "source_url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
    },
    {
        "name": "gemma-2-2b-it",
        "publisher": "google",
        "license": "Gemma Terms of Use",
        "license_url": "https://ai.google.dev/gemma/terms",
        "architecture": "gemma2",
        "parameter_count": "2B",
        "context_length": 8192,
        "quantization": "Q4_K_M",
        "ram_gb": 3.0,
        "vram_gb": 2.5,
        "download_size_gb": 1.7,
        "capabilities": ["chat"],
        "source_url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
    },
    {
        "name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "publisher": "deepseek-ai",
        "license": "MIT",
        "license_url": "https://opensource.org/licenses/MIT",
        "architecture": "qwen2",
        "parameter_count": "1.5B",
        "context_length": 131072,
        "quantization": "Q4_K_M",
        "ram_gb": 2.5,
        "vram_gb": 2.0,
        "download_size_gb": 1.1,
        "capabilities": ["reasoning", "chat"],
        "source_url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
    },
    {
        "name": "TinyLlama-1.1B-Chat-v1.0",
        "publisher": "TinyLlama",
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "architecture": "llama",
        "parameter_count": "1.1B",
        "context_length": 2048,
        "quantization": "Q4_K_M",
        "ram_gb": 2.0,
        "vram_gb": 1.5,
        "download_size_gb": 0.7,
        "capabilities": ["chat"],
        "channel": "stable",
        "source_url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    },
]


def _entry_to_record(entry: Dict[str, Any], *, origin: str = "bundled") -> ModelRecord:
    req = ModelRequirements(
        ram_gb=float(entry.get("ram_gb", 0.0) or 0.0),
        vram_gb=float(entry.get("vram_gb", 0.0) or 0.0),
        download_size_gb=float(entry.get("download_size_gb", 0.0) or 0.0),
        context_length=int(entry.get("context_length", 0) or 0),
        requires_gpu=bool(entry.get("requires_gpu", False)),
    )
    channel = entry.get("channel", Channel.STABLE.value)
    try:
        channel_enum = Channel(channel) if not isinstance(channel, Channel) else channel
    except ValueError:
        channel_enum = Channel.STABLE
    record = ModelRecord(
        name=str(entry.get("name", "")).strip(),
        publisher=str(entry.get("publisher", "")),
        source=str(entry.get("source", "catalog")),
        source_url=str(entry.get("source_url", "")),
        revision=str(entry.get("revision", "")),
        license=str(entry.get("license", "")),
        license_url=str(entry.get("license_url", "")),
        quantization=str(entry.get("quantization", "")),
        architecture=str(entry.get("architecture", "")),
        parameter_count=str(entry.get("parameter_count", "")),
        context_length=int(entry.get("context_length", 0) or 0),
        download_size_gb=float(entry.get("download_size_gb", 0.0) or 0.0),
        requirements=req,
        capabilities=[str(c) for c in (entry.get("capabilities") or [])],
        runtime=str(entry.get("runtime", "llama.cpp")),
        format=str(entry.get("format", "gguf")),
        sha256=str(entry.get("sha256", "") or ""),
        state=ModelState.DISCOVERED,
        channel=channel_enum,
    )
    extra = dict(entry.get("extra", {}) or {})
    extra.setdefault("catalog", origin)
    extra.setdefault("size_is_estimate", True)
    record.extra = extra
    record.discovered_at = float(entry.get("discovered_at", 0) or 0) or time.time()
    return record


def bundled_catalog() -> List[ModelRecord]:
    """Fresh (caller-owned) copies of the curated bundled catalog."""
    return [_entry_to_record(entry, origin="bundled") for entry in _BUNDLED]


# ---------------------------------------------------------------------------
# Catalog loading / merging
# ---------------------------------------------------------------------------


def catalog_to_data(records: Sequence[ModelRecord]) -> Dict[str, Any]:
    """Serialise records into the catalog document shape."""
    return {
        "version": CATALOG_VERSION,
        "generated_at": time.time(),
        "count": len(records),
        "models": [r.to_dict() for r in records],
    }


def catalog_from_data(data: Any, *, origin: str = "catalog") -> List[ModelRecord]:
    """Parse a catalog document (or a bare list of records), tolerantly."""
    if isinstance(data, dict):
        entries = data.get("models") or data.get("records") or []
    elif isinstance(data, list):
        entries = data
    else:
        return []
    records: List[ModelRecord] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        try:
            record = ModelRecord.from_dict(entry)
        except Exception:  # noqa: BLE001 - skip one bad row, keep the rest
            continue
        record.extra.setdefault("catalog", origin)
        records.append(record)
    return records


def load_catalog_file(path: str) -> List[ModelRecord]:
    """Load one catalog file; missing/corrupt files yield ``[]`` (never raise)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    return catalog_from_data(data, origin=os.path.basename(path))


def load_catalog(
    path: Optional[str] = None,
    *,
    include_bundled: bool = True,
    extra_paths: Iterable[str] = (),
) -> List[ModelRecord]:
    """Return the effective catalog.

    Order (lowest priority first): bundled entries, then the operator catalog
    at ``path`` (default :func:`catalog_path` if it exists), then any
    ``extra_paths``, then ``$HERMES_OFFLINE_CATALOG``.  Later definitions win
    for the same :attr:`ModelRecord.key`.
    """
    catalogs: List[List[ModelRecord]] = []
    if include_bundled:
        catalogs.append(bundled_catalog())

    candidate_paths: List[str] = []
    if path:
        candidate_paths.append(path)
    elif os.path.exists(catalog_path()):
        candidate_paths.append(catalog_path())
    candidate_paths.extend(p for p in extra_paths if p)
    env_path = os.environ.get("HERMES_OFFLINE_CATALOG")
    if env_path:
        candidate_paths.append(env_path)

    for candidate in candidate_paths:
        catalogs.append(load_catalog_file(candidate))
    return merge_catalogs(*catalogs)


def merge_catalogs(*catalogs: Iterable[ModelRecord]) -> List[ModelRecord]:
    """Merge record iterables by key; later entries override earlier ones."""
    merged: "dict[str, ModelRecord]" = {}
    for catalog in catalogs:
        for record in catalog:
            if record is None or not record.name:
                continue
            merged[record.key] = record
    return list(merged.values())


def save_catalog(records: Sequence[ModelRecord], path: Optional[str] = None) -> str:
    """Atomically write a catalog file; returns the path written."""
    target = path or catalog_path()
    directory = os.path.dirname(os.path.abspath(target))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{target}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(catalog_to_data(records), fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)
    return target


# ---------------------------------------------------------------------------
# Discovery / recommendation
# ---------------------------------------------------------------------------


def _capability_match(record: ModelRecord, wanted: Sequence[str]) -> bool:
    if not wanted:
        return True
    have = {c.lower() for c in record.capabilities}
    if "tools" in {w.lower() for w in wanted} and bool(record.extra.get("tools", False)):
        have.add("tools")
    return all(w.lower() in have for w in wanted)


def discover(
    registry: Optional[Any] = None,
    *,
    catalog: Optional[Sequence[ModelRecord]] = None,
    capabilities: Sequence[str] = (),
    runtime: str = "",
    max_download_gb: Optional[float] = None,
    hardware: Optional[HardwareProfile] = None,
    require_fit: bool = False,
    include_registry: bool = True,
    states: Sequence[ModelState] = (),
    limit: Optional[int] = None,
) -> List[ModelRecord]:
    """Discover candidate models from the catalog and local registry.

    Registry records (things actually downloaded/imported) take precedence over
    catalog metadata for the same key, so a real install's path/sha256/state are
    what get reported.  Filters are applied *after* the merge.
    """
    records: List[ModelRecord] = list(catalog) if catalog is not None else load_catalog()
    if include_registry and registry is not None:
        try:
            installed = registry.all()
        except Exception:  # noqa: BLE001
            installed = []
        records = merge_catalogs(records, installed)

    state_filter = {s if isinstance(s, ModelState) else ModelState(s) for s in states}
    out: List[ModelRecord] = []
    for record in records:
        if state_filter and record.state not in state_filter:
            continue
        if capabilities and not _capability_match(record, capabilities):
            continue
        if runtime and record.runtime and runtime.lower() not in record.runtime.lower():
            continue
        size = record.download_size_gb or record.disk_usage_gb or 0.0
        if max_download_gb is not None and size > max_download_gb:
            continue
        if hardware is not None:
            ok, reason = hardware.fits(record.requirements)
            if require_fit and not ok:
                continue
            record.extra["fit"] = bool(ok)
            record.extra["fit_reason"] = reason
        out.append(record)

    out.sort(key=lambda r: (r.publisher, r.name))
    if limit is not None:
        out = out[: max(0, int(limit))]
    return out


def recommend(
    hardware: HardwareProfile,
    *,
    catalog: Optional[Sequence[ModelRecord]] = None,
    capabilities: Sequence[str] = (),
    limit: int = 3,
) -> List[ModelRecord]:
    """Rank models that actually fit ``hardware`` for the wanted capabilities.

    Ranking prefers: capability coverage, then larger (more capable) models
    that still fit, then GPU usage.  Models that do not fit are excluded — we
    never recommend something the machine cannot run.
    """
    candidates = discover(
        catalog=catalog,
        capabilities=capabilities,
        hardware=hardware,
        require_fit=True,
        include_registry=False,
    )

    def score(record: ModelRecord) -> "tuple[float, float, float]":
        coverage = len(set(c.lower() for c in record.capabilities) & {c.lower() for c in capabilities})
        size = record.download_size_gb or 0.0
        vram_fit = 1.0 if (hardware.has_gpu and record.requirements.vram_gb) else 0.0
        return (float(coverage), size, vram_fit)

    candidates.sort(key=score, reverse=True)
    return candidates[: max(0, int(limit))]


__all__ = [
    "bundled_catalog",
    "catalog_from_data",
    "catalog_path",
    "catalog_to_data",
    "discover",
    "load_catalog",
    "load_catalog_file",
    "merge_catalogs",
    "offline_home",
    "recommend",
    "save_catalog",
    "CATALOG_FILENAME",
    "CATALOG_VERSION",
]
