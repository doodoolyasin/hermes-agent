"""Real hardware / resource profiling (stdlib only).

Reads real values from ``/proc``, ``/sys``, ``platform``, ``shutil`` and
optional vendor CLIs (``nvidia-smi``, ``rocm-smi``).  Every probe is defensive:
a failure adds a note instead of raising, and unknown values stay ``0``/``""``
rather than being invented.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Optional

from .models import HardwareProfile, ModelRequirements

__all__ = ["gb", "detect", "summarise", "PROC_MEMINFO", "PROC_CPUINFO"]

PROC_MEMINFO = "/proc/meminfo"
PROC_CPUINFO = "/proc/cpuinfo"


def gb(num_bytes: float) -> float:
    """Bytes → GiB, rounded to 2 decimals."""
    try:
        return round(float(num_bytes) / (1024 ** 3), 2)
    except (TypeError, ValueError):
        return 0.0


def _read_first(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _meminfo() -> dict:
    out: dict = {}
    for line in _read_first(PROC_MEMINFO).splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        val = rest.strip().split()
        if not val:
            continue
        try:
            out[key.strip()] = float(val[0]) * 1024.0  # kB → bytes
        except (TypeError, ValueError):
            continue
    return out


def _cpu_model() -> str:
    for line in _read_first(PROC_CPUINFO).splitlines():
        if line.lower().startswith(("model name", "hardware", "cpu model")):
            _, _, val = line.partition(":")
            val = val.strip()
            if val:
                return val
    return platform.processor() or ""


def _run(cmd: list, timeout: float = 5.0) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _detect_gpu(profile: HardwareProfile) -> None:
    # NVIDIA
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits"])
    if out:
        try:
            name, total, free = [p.strip() for p in out.splitlines()[0].split(",")]
            profile.has_gpu = True
            profile.gpu_model = name
            profile.vram_total_gb = round(float(total) / 1024.0, 2)
            profile.vram_available_gb = round(float(free) / 1024.0, 2)
            if "cuda" not in profile.accelerators:
                profile.accelerators.append("cuda")
            return
        except Exception:
            profile.notes.append("nvidia-smi output could not be parsed")

    # AMD ROCm
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if out:
        profile.notes.append("rocm-smi detected; VRAM parsing is best-effort")
        profile.has_gpu = True
        if "rocm" not in profile.accelerators:
            profile.accelerators.append("rocm")
        return

    # macOS Metal
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        profile.has_gpu = True
        profile.gpu_model = profile.gpu_model or "Apple Silicon (Metal)"
        if "metal" not in profile.accelerators:
            profile.accelerators.append("metal")


def detect(download_dir: Optional[str] = None) -> HardwareProfile:
    """Return a :class:`HardwareProfile` populated from real system probes."""
    profile = HardwareProfile()
    profile.os_name = platform.system()
    profile.os_version = platform.release()
    profile.arch = platform.machine()

    try:
        profile.cpu_model = _cpu_model()
    except Exception as exc:  # noqa: BLE001
        profile.notes.append(f"cpu model probe failed: {type(exc).__name__}")

    cores = os.cpu_count() or 0
    profile.cpu_cores = cores
    profile.cpu_threads = cores
    if not cores:
        profile.notes.append("cpu core count unavailable")

    mem = _meminfo()
    if mem.get("MemTotal"):
        profile.ram_total_gb = gb(mem["MemTotal"])
    else:
        profile.notes.append("MemTotal unavailable (non-Linux?)")
    if mem.get("MemAvailable"):
        profile.ram_available_gb = gb(mem["MemAvailable"])
    elif profile.ram_total_gb:
        profile.ram_available_gb = profile.ram_total_gb  # best-effort upper bound

    target = download_dir or os.path.expanduser("~") or "."
    try:
        usage = shutil.disk_usage(target)
        profile.disk_free_gb = gb(usage.free)
        profile.disk_total_gb = gb(usage.total)
    except Exception as exc:  # noqa: BLE001
        profile.notes.append(f"disk probe failed for {target}: {type(exc).__name__}")

    try:
        _detect_gpu(profile)
    except Exception as exc:  # noqa: BLE001
        profile.notes.append(f"gpu probe failed: {type(exc).__name__}")

    if "cpu" not in profile.accelerators:
        profile.accelerators.append("cpu")
    return profile


def summarise(profile: HardwareProfile) -> str:
    """One-line human summary for CLI / doctor output."""
    accel = ", ".join(profile.accelerators) or "cpu"
    gpu = f"{profile.gpu_model} ({profile.vram_total_gb}GB VRAM)" if profile.has_gpu else "no GPU"
    return (
        f"{profile.os_name} {profile.os_version} / {profile.arch} | "
        f"{profile.cpu_cores} cores | RAM {profile.ram_available_gb}/{profile.ram_total_gb}GB | "
        f"disk free {profile.disk_free_gb}GB | {gpu} | accel: {accel}"
    )
