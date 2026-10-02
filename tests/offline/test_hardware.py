"""Tests for hermes_offline.hardware (real, defensive profiling)."""

from __future__ import annotations

from hermes_offline import hardware
from hermes_offline.models import HardwareProfile, ModelRequirements


def test_gb_conversion():
    assert hardware.gb(1024 ** 3) == 1.0
    assert hardware.gb(0) == 0.0
    assert hardware.gb("bad") == 0.0


def test_detect_returns_sane_values():
    profile = hardware.detect()
    assert isinstance(profile, HardwareProfile)
    assert profile.os_name
    assert profile.arch
    assert profile.cpu_cores >= 0
    assert profile.ram_total_gb >= 0.0
    assert profile.disk_free_gb >= 0.0
    assert profile.disk_total_gb >= 0.0
    assert "cpu" in profile.accelerators


def test_detect_is_defensive_when_proc_unreadable(monkeypatch):
    monkeypatch.setattr(hardware, "_read_first", lambda path: "")
    profile = hardware.detect()
    assert isinstance(profile, HardwareProfile)
    assert profile.ram_total_gb == 0.0
    assert any("MemTotal" in n for n in profile.notes)


def test_detect_handles_disk_failure(monkeypatch):
    def boom(_path):
        raise OSError("nope")

    monkeypatch.setattr(hardware.shutil, "disk_usage", boom)
    profile = hardware.detect()
    assert profile.disk_free_gb == 0.0
    assert any("disk probe failed" in n for n in profile.notes)


def test_summarise_mentions_core_facts():
    profile = HardwareProfile(os_name="Linux", arch="x86_64", cpu_cores=8,
                              ram_total_gb=16.0, ram_available_gb=8.0, disk_free_gb=50.0)
    text = hardware.summarise(profile)
    assert "Linux" in text and "8 cores" in text


# ── HardwareProfile.fits ─────────────────────────────────────────────
def test_fits_cpu_model_with_enough_ram():
    profile = HardwareProfile(ram_total_gb=32.0, ram_available_gb=24.0, disk_free_gb=100.0)
    ok, reason = profile.fits(ModelRequirements(ram_gb=8.0, download_size_gb=5.0))
    assert ok is True
    assert "ram" in reason.lower()  # informative reason is still returned


def test_fits_rejects_insufficient_ram():
    profile = HardwareProfile(ram_total_gb=8.0, ram_available_gb=2.0, disk_free_gb=100.0)
    ok, reason = profile.fits(ModelRequirements(ram_gb=16.0, download_size_gb=5.0))
    assert ok is False and "ram" in reason.lower()


def test_fits_rejects_when_gpu_required_but_absent():
    profile = HardwareProfile(ram_total_gb=64.0, ram_available_gb=64.0, disk_free_gb=100.0,
                              has_gpu=False)
    ok, reason = profile.fits(ModelRequirements(vram_gb=8.0, requires_gpu=True,
                                                download_size_gb=5.0))
    assert ok is False and ("gpu" in reason.lower() or "vram" in reason.lower())


def test_fits_rejects_insufficient_disk():
    profile = HardwareProfile(ram_total_gb=32.0, ram_available_gb=32.0, disk_free_gb=1.0)
    ok, reason = profile.fits(ModelRequirements(ram_gb=2.0, download_size_gb=10.0))
    assert ok is False and "disk" in reason.lower()


def test_fits_uses_vram_when_gpu_present():
    profile = HardwareProfile(ram_total_gb=8.0, ram_available_gb=4.0, disk_free_gb=100.0,
                              has_gpu=True, vram_total_gb=24.0, vram_available_gb=20.0)
    ok, _ = profile.fits(ModelRequirements(vram_gb=8.0, ram_gb=16.0, requires_gpu=True,
                                           download_size_gb=5.0))
    assert ok is True
