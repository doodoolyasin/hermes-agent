"""Shared data contracts for :mod:`hermes_offline`.

Everything here is a plain dataclass / enum with ``to_dict`` support so it can
be serialised into the local registry, JSON diagnostics, or the readiness
report without pulling in a YAML/JSON schema library.

Standard library only — this module is imported by the emergency tooling and
must never fail to import.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


class ConnectivityState(str, Enum):
    """Coarse-grained connectivity state.

    The distinction between these states matters: treating every failure as
    "offline" is exactly the bug this subsystem exists to avoid.
    """

    FULL_ONLINE = "full_online"
    DEGRADED = "degraded"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    OFFLINE = "offline"
    EMERGENCY_OFFLINE_READY = "emergency_offline_ready"

    def __str__(self) -> str:  # nicer logging / CLI output
        return self.value


class FailureClass(str, Enum):
    """How a network operation failed — used to classify *why* we are degraded."""

    NONE = "none"
    DNS = "dns"
    TLS = "tls"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    HTTP_4XX = "http_4xx"
    HTTP_5XX = "http_5xx"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"


@dataclass
class ProbeResult:
    """Result of a single endpoint reachability probe."""

    endpoint: str
    ok: bool
    latency_ms: Optional[float] = None
    failure: FailureClass = FailureClass.NONE
    status_code: Optional[int] = None
    detail: str = ""
    checked_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["failure"] = self.failure.value
        return d


# ---------------------------------------------------------------------------
# Model / hardware contracts
# ---------------------------------------------------------------------------


class Channel(str, Enum):
    """Release channel for a model artifact."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"


class ModelState(str, Enum):
    """Lifecycle state of a model in the local registry."""

    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    INSTALLED = "installed"
    VERIFIED = "verified"
    FAILED = "failed"


class DownloadState(str, Enum):
    """Lifecycle state of a single download attempt."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    INSUFFICIENT_DISK = "insufficient_disk"


@dataclass
class ModelRequirements:
    """Resource requirements of a model, in (fractional) gigabytes."""

    ram_gb: float = 0.0
    vram_gb: float = 0.0
    download_size_gb: float = 0.0
    context_length: int = 0
    requires_gpu: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class HardwareProfile:
    """Detected hardware / resource profile.

    ``fits()`` deliberately applies headroom and considers both RAM and VRAM:
    a model that merely fits on disk is not necessarily *usable*.
    """

    os_name: str = ""
    os_version: str = ""
    arch: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0

    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0

    has_gpu: bool = False
    gpu_model: str = ""
    vram_total_gb: float = 0.0
    vram_available_gb: float = 0.0

    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0

    accelerators: List[str] = field(default_factory=list)  # e.g. ["cuda", "rocm", "metal", "cpu"]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def fits(self, req: ModelRequirements, *, headroom: float = 1.25) -> "tuple[bool, str]":
        """Return ``(ok, reason)`` for whether ``req`` is realistically usable.

        ``headroom`` is the safety multiplier applied to memory requirements
        (default 1.25 = 25% spare) so we do not recommend a model that only
        *just* fits and then thrashes under real runtime pressure.
        """
        # 1) Disk: the artifact must fit with a little slack.
        if req.download_size_gb > 0 and self.disk_free_gb < req.download_size_gb * 1.05:
            return False, (
                f"insufficient disk: need {req.download_size_gb:.1f} GB, "
                f"have {self.disk_free_gb:.1f} GB free"
            )

        need = req.vram_gb * headroom if req.vram_gb else 0.0
        if req.requires_gpu and not self.has_gpu:
            return False, "model requires a GPU but none was detected"

        # 2) Prefer VRAM when a GPU is present and the model declares one.
        if self.has_gpu and req.vram_gb > 0:
            if self.vram_total_gb >= need:
                return True, (
                    f"fits in VRAM ({self.vram_total_gb:.1f} GB total, "
                    f"need {need:.1f} GB incl. headroom)"
                )

        # 3) Otherwise fall back to system RAM (CPU / partial offload).
        ram_need = req.ram_gb * headroom if req.ram_gb else 0.0
        if ram_need and self.ram_total_gb >= ram_need:
            # Available (not total) RAM decides practical usability.
            if self.ram_available_gb and self.ram_available_gb < ram_need * 0.9:
                return False, (
                    f"RAM pressure: {self.ram_available_gb:.1f} GB available now, "
                    f"need ~{ram_need:.1f} GB"
                )
            return True, f"fits in RAM (need ~{ram_need:.1f} GB incl. headroom)"

        if ram_need:
            return False, (
                f"insufficient memory: need ~{ram_need:.1f} GB RAM, "
                f"have {self.ram_total_gb:.1f} GB total"
            )
        return True, "no explicit requirements declared"


@dataclass
class ModelRecord:
    """One model artifact tracked by the local registry.

    Field set follows the discovery/metadata requirements; ``extra`` holds
    source-specific fields without forcing a schema change.
    """

    name: str
    publisher: str = ""
    source: str = ""  # e.g. "huggingface", "ollama", "import", "catalog"
    source_url: str = ""
    revision: str = ""
    license: str = ""
    license_url: str = ""

    quantization: str = ""
    architecture: str = ""
    parameter_count: str = ""
    context_length: int = 0

    download_size_gb: float = 0.0
    requirements: ModelRequirements = field(default_factory=ModelRequirements)

    capabilities: List[str] = field(default_factory=list)  # tools, vision, multilingual, coding, reasoning
    runtime: str = ""  # preferred local runtime ("ollama", "llama.cpp", ...)
    format: str = ""   # gguf, safetensors, ...

    sha256: str = ""
    local_path: str = ""

    state: ModelState = ModelState.DISCOVERED
    channel: Channel = Channel.STABLE
    fail_count: int = 0
    last_error: str = ""

    discovered_at: Optional[float] = None
    last_checked: Optional[float] = None
    installed_at: Optional[float] = None
    last_verified: Optional[float] = None

    disk_usage_gb: float = 0.0
    pinned: bool = False  # never prune a pinned (known-good) model

    extra: Dict[str, Any] = field(default_factory=dict)

    # -- helpers -----------------------------------------------------------

    @property
    def key(self) -> str:
        """Stable identity: publisher/name@revision (revision optional)."""
        base = f"{self.publisher}/{self.name}" if self.publisher else self.name
        return f"{base}@{self.revision}" if self.revision else base

    @property
    def is_ready(self) -> bool:
        """A model is usable only once installed *and* verified."""
        return self.state in (ModelState.VERIFIED,) and bool(self.local_path)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["state"] = self.state.value
        d["channel"] = self.channel.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRecord":
        data = dict(data)
        req = data.pop("requirements", None) or {}
        if isinstance(req, dict):
            data["requirements"] = ModelRequirements(**{
                k: v for k, v in req.items()
                if k in ModelRequirements.__dataclass_fields__
            })
        for enum_field, enum_cls in (("state", ModelState), ("channel", Channel)):
            val = data.get(enum_field)
            if isinstance(val, str):
                try:
                    data[enum_field] = enum_cls(val)
                except ValueError:
                    data[enum_field] = enum_cls.__members__[enum_field.upper()] \
                        if enum_field.upper() in enum_cls.__members__ else list(enum_cls)[0]
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})


# ---------------------------------------------------------------------------
# Runtime / readiness
# ---------------------------------------------------------------------------


@dataclass
class RuntimeInfo:
    """State of one local inference runtime."""

    name: str
    available: bool = False
    version: str = ""
    endpoint: str = ""
    models: List[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class ReadinessLevel(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"


@dataclass
class ReadinessItem:
    """One line of the emergency readiness report — computed from a REAL check."""

    name: str
    level: ReadinessLevel
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.level in (ReadinessLevel.READY, ReadinessLevel.DEGRADED)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "level": self.level.value, "detail": self.detail}


@dataclass
class ReadinessReport:
    """Aggregate technical readiness report (not a subjective score)."""

    items: List[ReadinessItem] = field(default_factory=list)
    connectivity: ConnectivityState = ConnectivityState.FULL_ONLINE
    generated_at: Optional[float] = None

    @property
    def ready(self) -> bool:
        """Emergency mode is ready when every *required* item is at least DEGRADED."""
        required = {"local_model", "runtime", "tools", "memory", "offline_docs", "recovery", "disk"}
        return all(
            i.ok for i in self.items if i.name in required
        ) and bool(self.items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "connectivity": self.connectivity.value,
            "generated_at": self.generated_at,
            "items": [i.to_dict() for i in self.items],
        }

    def render(self) -> str:
        """Human-readable table used by ``hermes offline status`` / doctor."""
        lines = ["HERMES EMERGENCY READINESS", ""]
        width = max((len(i.name) for i in self.items), default=12)
        for item in self.items:
            label = item.name.replace("_", " ").title().ljust(width)
            mark = item.level.value.upper()
            line = f"{label}   {mark}"
            if item.detail:
                line += f"   ({item.detail})"
            lines.append(line)
        lines.append("")
        lines.append(f"Emergency Mode    {'READY' if self.ready else 'NOT READY'}")
        return "\n".join(lines)
