"""Resumable, checksum-verified artifact downloads.

The downloader exists so that a large model artifact can be fetched over an
unreliable / throttled link and still end up *provably* correct, or not at all:

* Data streams into ``<file>.part``; the completed artifact is only moved to
  its final name after its SHA-256 matches the expected value
  (``os.replace`` is atomic on POSIX and Windows).
* Resuming uses HTTP ``Range`` against the existing ``.part`` file.  A server
  that ignores ``Range`` (200 instead of 206) transparently restarts the
  transfer rather than corrupting the partial file.
* Free disk space is checked *before* writing anything.
* A checksum mismatch never deletes the evidence: the bad partial is
  quarantined as ``<file>.part.bad-<timestamp>``.
* ``file://`` URLs and bare local paths are supported, so an air-gapped
  machine can "download" from a mounted USB stick or local cache with the same
  code path and the same verification guarantees.

Standard library only (``urllib``, ``hashlib``, ``shutil``).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .models import DownloadState

#: Chunk size used when hashing files and reading streams.
DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB
#: Default per-request timeout in seconds.
DEFAULT_TIMEOUT = 30.0
#: Free-space slack multiplier applied to a known artifact size.
DISK_HEADROOM = 1.02

ProgressCallback = Callable[[int, int], None]


def default_models_dir() -> str:
    """``$HERMES_HOME/offline/models`` without importing ``emergency``."""
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(home, "offline", "models")


class DownloadError(Exception):
    """Raised for unrecoverable downloader misuse (not a transient failure)."""


def sha256_file(path: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the hex SHA-256 of ``path`` (raises :class:`OSError` if unreadable)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    expected: Optional[str] = None,
) -> Optional[str]:
    """Compute the SHA-256 hex digest of ``path``.

    Returns ``None`` when the file cannot be read.  When ``expected`` is given
    the digest is still returned (the caller decides what a mismatch means), so
    callers can report both values instead of a bare boolean.
    """
    try:
        return sha256_file(path, chunk_size=chunk_size)
    except OSError:
        return None


@dataclass
class DownloadSpec:
    """Everything needed to fetch and verify one artifact."""

    name: str = ""
    url: str = ""
    sha256: str = ""
    dest_dir: str = ""
    filename: str = ""
    size_bytes: int = 0
    sources: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    chunk_size: int = DEFAULT_CHUNK_SIZE
    timeout: float = DEFAULT_TIMEOUT
    resume: bool = True

    def all_sources(self) -> List[str]:
        """Primary URL followed by mirrors, de-duplicated, order preserved."""
        ordered: List[str] = []
        for candidate in [self.url, *(self.sources or [])]:
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        return ordered

    def resolved_filename(self) -> str:
        if self.filename:
            return self.filename
        if self.url:
            base = os.path.basename(urllib.parse.urlparse(self.url).path)
            if base:
                return base
        return self.name or "model.bin"

    def dest_path(self, base_dir: Optional[str] = None) -> str:
        directory = self.dest_dir or base_dir or default_models_dir()
        return os.path.join(directory, self.resolved_filename())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "sha256": self.sha256,
            "dest_dir": self.dest_dir,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "sources": list(self.sources),
            "chunk_size": self.chunk_size,
        }


@dataclass
class DownloadResult:
    """Outcome of a single :meth:`Downloader.download` call."""

    spec: Optional[DownloadSpec] = None
    state: DownloadState = DownloadState.PENDING
    path: str = ""
    part_path: str = ""
    sha256: str = ""
    bytes_downloaded: int = 0
    total_bytes: int = 0
    resumed: bool = False
    attempts: int = 0
    source: str = ""
    error: str = ""
    quarantine_path: str = ""
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.state == DownloadState.COMPLETE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": getattr(self.spec, "name", ""),
            "state": self.state.value,
            "path": self.path,
            "sha256": self.sha256,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "resumed": self.resumed,
            "attempts": self.attempts,
            "source": self.source,
            "error": self.error,
            "quarantine_path": self.quarantine_path,
            "duration_s": round(self.duration_s, 3),
        }


class Downloader:
    """Resumable, checksum-verified downloader.

    Construct with no arguments for the defaults used by the CLI; every knob is
    overridable so tests and callers can point it at a local mirror or a temp
    directory.
    """

    def __init__(
        self,
        *,
        dest_dir: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = 3,
        backoff: float = 0.5,
        allow_network: bool = True,
        proxy: Optional[str] = None,
        mirrors: Optional[List[str]] = None,
        user_agent: str = "hermes-offline/1.0",
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.dest_dir = dest_dir
        self.chunk_size = max(4096, int(chunk_size))
        self.timeout = float(timeout)
        self.retries = max(1, int(retries))
        self.backoff = max(0.0, float(backoff))
        self.allow_network = bool(allow_network)
        self.proxy = proxy if proxy is not None else (os.environ.get("HERMES_OFFLINE_PROXY") or None)
        if mirrors is None:
            env = os.environ.get("HERMES_OFFLINE_MIRRORS", "")
            mirrors = [m.strip() for m in env.split(",") if m.strip()]
        self.mirrors = list(mirrors or [])
        self.user_agent = user_agent
        self.progress = progress

    # -- public API --------------------------------------------------------

    def download(self, spec: DownloadSpec) -> DownloadResult:
        started = time.monotonic()
        result = DownloadResult(spec=spec, state=DownloadState.PENDING)
        base_dir = spec.dest_dir or self.dest_dir or default_models_dir()
        try:
            os.makedirs(base_dir, exist_ok=True)
        except OSError as exc:
            result.state = DownloadState.FAILED
            result.error = f"cannot create destination directory {base_dir!r}: {exc}"
            result.duration_s = time.monotonic() - started
            return result

        dest = spec.dest_path(base_dir)
        part = dest + ".part"
        result.path = dest
        result.part_path = part

        if not spec.name and not spec.url:
            result.state = DownloadState.FAILED
            result.error = "download spec has neither a name nor a url"
            result.duration_s = time.monotonic() - started
            return result

        try:
            state, error, quarantine = self._download_into(spec, dest, part, result)
        except Exception as exc:  # noqa: BLE001 - a bad source must not crash the pack
            state, error, quarantine = DownloadState.FAILED, f"{type(exc).__name__}: {exc}", ""
        result.state = state
        result.error = error
        result.quarantine_path = quarantine
        result.duration_s = time.monotonic() - started
        return result

    def download_many(self, specs: List[DownloadSpec]) -> List[DownloadResult]:
        return [self.download(spec) for spec in specs]

    def cleanup_partials(self, directory: Optional[str] = None) -> List[str]:
        """Remove stray ``.part`` files; returns the paths removed."""
        target = directory or self.dest_dir or default_models_dir()
        removed: List[str] = []
        try:
            names = os.listdir(target)
        except OSError:
            return removed
        for name in names:
            if name.endswith(".part"):
                path = os.path.join(target, name)
                try:
                    os.remove(path)
                    removed.append(path)
                except OSError:
                    pass
        return removed

    # -- orchestration -----------------------------------------------------

    def _download_into(
        self,
        spec: DownloadSpec,
        dest: str,
        part: str,
        result: DownloadResult,
    ) -> "tuple[DownloadState, str, str]":
        sources = spec.all_sources()
        # ``HERMES_OFFLINE_MIRRORS`` holds *base* URLs, so resolve each against
        # the artifact filename rather than treating it as a full URL.
        filename = spec.resolved_filename()
        for mirror in self.mirrors:
            resolved = _join_mirror(mirror, filename)
            if resolved and resolved not in sources:
                sources.append(resolved)

        # Idempotency: an already-correct artifact needs no work.
        if os.path.exists(dest) and not os.path.exists(part):
            if spec.sha256:
                digest = verify_file(dest, chunk_size=self.chunk_size)
                if digest and digest.lower() == spec.sha256.lower():
                    result.sha256 = digest
                    result.bytes_downloaded = _safe_size(dest)
                    result.total_bytes = spec.size_bytes or result.bytes_downloaded
                    return DownloadState.COMPLETE, "", ""
            else:
                result.sha256 = verify_file(dest, chunk_size=self.chunk_size) or ""
                result.bytes_downloaded = _safe_size(dest)
                result.total_bytes = result.bytes_downloaded
                return DownloadState.COMPLETE, "", ""
            # Present but wrong (or unverifiable against a given hash): set aside.
            existing = self._quarantine(dest)
            result.quarantine_path = existing

        if not sources:
            return DownloadState.FAILED, "no source URL and no mirror configured", ""

        # Disk preflight before writing a single byte.
        have = _safe_size(part)
        remaining = max(spec.size_bytes - have, 0) if spec.size_bytes > 0 else 0
        if spec.size_bytes > 0 and not self._has_space(dest_dir_of(dest), remaining + self.chunk_size):
            return (
                DownloadState.INSUFFICIENT_DISK,
                f"need {remaining / (1024 ** 3):.2f} GB more free space for {os.path.basename(dest)}",
                "",
            )

        last_error = ""
        for source in sources:
            for attempt in range(1, self.retries + 1):
                result.attempts += 1
                result.source = source
                try:
                    fetched = self._fetch_one(spec, source, part, result)
                except _TransientFailure as exc:
                    last_error = f"{source}: {exc}"
                    if attempt < self.retries:
                        time.sleep(self.backoff * attempt)
                        continue
                    break
                except _IntegrityFailure as exc:
                    # Deterministic: quarantining the partial is the honest move.
                    quarantine = self._quarantine(part)
                    result.quarantine_path = quarantine
                    return DownloadState.CHECKSUM_MISMATCH, str(exc), quarantine
                except _PermanentFailure as exc:
                    last_error = f"{source}: {exc}"
                    break
                # Transfer finished; verify.
                if spec.size_bytes > 0:
                    got = _safe_size(part)
                    if got != spec.size_bytes:
                        last_error = (
                            f"{source}: incomplete transfer "
                            f"({got} of {spec.size_bytes} bytes)"
                        )
                        if attempt < self.retries:
                            time.sleep(self.backoff * attempt)
                            continue
                        break
                if spec.sha256:
                    digest = verify_file(part, chunk_size=self.chunk_size)
                    if digest is None:
                        last_error = f"{source}: downloaded file vanished before verification"
                        break
                    if digest.lower() != spec.sha256.lower():
                        quarantine = self._quarantine(part)
                        result.quarantine_path = quarantine
                        return (
                            DownloadState.CHECKSUM_MISMATCH,
                            f"sha256 mismatch: expected {spec.sha256[:12]}…, got {digest[:12]}…",
                            quarantine,
                        )
                    result.sha256 = digest
                else:
                    result.sha256 = verify_file(part, chunk_size=self.chunk_size) or ""

                # Atomic completion: only now does the artifact get its real name.
                try:
                    os.replace(part, dest)
                except OSError as exc:
                    return DownloadState.FAILED, f"cannot finalize {dest!r}: {exc}", ""
                result.bytes_downloaded = _safe_size(dest)
                result.total_bytes = spec.size_bytes or result.bytes_downloaded
                return DownloadState.COMPLETE, "", ""
        return DownloadState.FAILED, last_error or "all sources failed", ""

    # -- per-source transports --------------------------------------------

    def _fetch_one(
        self,
        spec: DownloadSpec,
        source: str,
        part: str,
        result: DownloadResult,
    ) -> None:
        scheme = urllib.parse.urlparse(source).scheme.lower()
        if scheme in ("", "file"):
            self._fetch_file(spec, source, part, result)
            return
        if scheme in ("http", "https"):
            if not self.allow_network:
                raise _PermanentFailure("network access is disabled (--offline)")
            self._fetch_http(spec, source, part, result)
            return
        raise _PermanentFailure(f"unsupported URL scheme {scheme!r}")

    def _fetch_file(self, spec: DownloadSpec, source: str, part: str, result: DownloadResult) -> None:
        src = urllib.request.url2pathname(urllib.parse.urlparse(source).path) if source.startswith("file:") else source
        if not os.path.exists(src):
            raise _PermanentFailure(f"local source not found: {src}")
        src_size = os.path.getsize(src)
        offset = _safe_size(part) if spec.resume else 0
        if offset >= src_size:
            # Partial is already >= source: restart cleanly rather than trust it.
            offset = 0
        result.resumed = offset > 0
        total = src_size
        mode = "ab" if offset else "wb"
        written = offset
        with open(src, "rb") as reader, open(part, mode) as writer:
            if offset:
                reader.seek(offset)
            while True:
                chunk = reader.read(self.chunk_size)
                if not chunk:
                    break
                writer.write(chunk)
                written += len(chunk)
                self._emit_progress(written, total)
            writer.flush()
            os.fsync(writer.fileno())
        result.bytes_downloaded = written
        result.total_bytes = total

    def _fetch_http(self, spec: DownloadSpec, source: str, part: str, result: DownloadResult) -> None:
        offset = _safe_size(part) if spec.resume else 0
        headers = {"User-Agent": self.user_agent, "Accept": "*/*"}
        headers.update({k: v for k, v in (spec.headers or {}).items() if v is not None})
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"

        request = urllib.request.Request(source, headers=headers, method="GET")
        opener = self._opener()
        try:
            resp = opener.open(request, timeout=spec.timeout or self.timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and offset > 0:
                # Range not satisfiable: the partial may already be complete.
                result.resumed = True
                result.bytes_downloaded = offset
                return
            retryable = exc.code >= 500 or exc.code == 429
            raise (_TransientFailure if retryable else _PermanentFailure)(
                f"HTTP {exc.code} {exc.reason}"
            )
        except urllib.error.URLError as exc:
            raise _TransientFailure(f"connection error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise _TransientFailure(f"timeout: {exc}") from exc
        except OSError as exc:
            raise _TransientFailure(f"{type(exc).__name__}: {exc}") from exc

        with resp:
            status = getattr(resp, "status", 200)
            total = _content_total(resp, offset)
            if status == 206 and offset > 0:
                mode, written, result.resumed = "ab", offset, True
            else:
                # Server ignored the range (or we had none): start over cleanly.
                mode, written, result.resumed = "wb", 0, False
            with open(part, mode) as fh:
                while True:
                    chunk = resp.read(self.chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
                    self._emit_progress(written, total)
                fh.flush()
                os.fsync(fh.fileno())
            result.bytes_downloaded = written
            result.total_bytes = total

    def _opener(self) -> urllib.request.OpenerDirector:
        if self.proxy:
            handler = urllib.request.ProxyHandler({"http": self.proxy, "https": self.proxy})
            return urllib.request.build_opener(handler)
        return urllib.request.build_opener()

    # -- helpers -----------------------------------------------------------

    def _emit_progress(self, done: int, total: int) -> None:
        if self.progress is None:
            return
        try:
            self.progress(done, total)
        except Exception:  # noqa: BLE001 - progress must never break a download
            pass

    def _has_space(self, directory: str, needed_bytes: int) -> bool:
        try:
            free = shutil.disk_usage(directory).free
        except OSError:
            return True  # cannot tell -> don't block the attempt
        return free >= int(needed_bytes * DISK_HEADROOM)

    @staticmethod
    def _quarantine(path: str) -> str:
        dest = f"{path}.bad-{int(time.time())}"
        counter = 0
        while os.path.exists(dest):
            counter += 1
            dest = f"{path}.bad-{int(time.time())}-{counter}"
        try:
            os.replace(path, dest)
            return dest
        except OSError:
            return ""


# ---------------------------------------------------------------------------
# Small internal helpers / failure taxonomy
# ---------------------------------------------------------------------------


class _TransientFailure(Exception):
    """A failure worth retrying against the same source."""


class _PermanentFailure(Exception):
    """A failure that will not improve by retrying at this moment."""


class _IntegrityFailure(Exception):
    """The bytes arrived but are provably wrong."""


def dest_dir_of(path: str) -> str:
    return os.path.dirname(os.path.abspath(path)) or "."


def _join_mirror(base: str, filename: str) -> str:
    """Resolve ``filename`` against a mirror base URL (or local directory)."""
    if not base:
        return ""
    if base.endswith(("/", "\\")):
        return base + filename
    scheme = urllib.parse.urlparse(base).scheme.lower()
    if scheme in ("http", "https"):
        return base.rstrip("/") + "/" + filename
    return os.path.join(base, filename)


def _safe_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _content_total(resp: Any, offset: int) -> int:
    """Total artifact size from Content-Length / Content-Range, or 0 if unknown."""
    content_range = resp.getheader("Content-Range") if hasattr(resp, "getheader") else None
    if content_range and "/" in content_range:
        tail = content_range.rsplit("/", 1)[1].strip()
        if tail.isdigit():
            return int(tail)
    length = resp.getheader("Content-Length") if hasattr(resp, "getheader") else None
    if length and str(length).isdigit():
        return int(length) + (offset if getattr(resp, "status", 200) == 206 else 0)
    return 0


__all__ = [
    "DownloadSpec",
    "DownloadResult",
    "Downloader",
    "DownloadError",
    "sha256_file",
    "verify_file",
    "default_models_dir",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_TIMEOUT",
]
