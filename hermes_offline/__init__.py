"""Hermes offline / resilient-connectivity subsystem.

This package provides the building blocks for operating Hermes when
international connectivity is degraded or entirely unavailable:

* :mod:`hermes_offline.connectivity` — classified connectivity states,
  circuit breakers, bounded retry/backoff, endpoint + provider health.
* :mod:`hermes_offline.hardware` — real hardware/resource profiling.
* :mod:`hermes_offline.registry` — local model registry (JSON-backed).
* :mod:`hermes_offline.download` — resumable, checksum-verified downloads.
* :mod:`hermes_offline.discovery` — offline model discovery + metadata.
* :mod:`hermes_offline.runtimes` — local inference runtime adapters.
* :mod:`hermes_offline.emergency` — emergency pack prepare/verify/readiness.
* :mod:`hermes_offline.docs` — offline documentation generation.
* :mod:`hermes_offline.cli` — ``hermes offline ...`` subcommands.

Pure standard library only: this package must import with no third-party
dependency so it keeps working in an emergency.
"""

from hermes_offline.models import (  # noqa: F401
    Channel,
    ConnectivityState,
    DownloadState,
    FailureClass,
    HardwareProfile,
    ModelRecord,
    ModelRequirements,
    ModelState,
    ProbeResult,
    ReadinessItem,
    ReadinessReport,
    RuntimeInfo,
)

__all__ = [
    "Channel",
    "ConnectivityState",
    "DownloadState",
    "FailureClass",
    "HardwareProfile",
    "ModelRecord",
    "ModelRequirements",
    "ModelState",
    "ProbeResult",
    "ReadinessItem",
    "ReadinessReport",
    "RuntimeInfo",
]
