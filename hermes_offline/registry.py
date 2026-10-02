"""Local model registry backed by a single atomic JSON file.

The registry is the source of truth for what model artifacts this machine has
seen, downloaded, installed and verified.  It is deliberately dependency-free
and crash-safe:

* Every write goes to a unique temp file in the same directory, is ``fsync``-ed,
  then ``os.replace``-d onto the real path.  A crash mid-write can therefore
  never leave a truncated ``registry.json`` behind.
* A corrupt/unreadable registry is *quarantined* (renamed aside) rather than
  silently discarded or allowed to break the whole offline subsystem.
* State transitions always go through :class:`~hermes_offline.models.ModelState`
  so callers cannot record an impossible state like ``verified`` without a path.

Only the Python standard library is used.
"""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .models import Channel, ModelRecord, ModelState

#: Schema version for the on-disk JSON document.
REGISTRY_VERSION = 1


def default_registry_path() -> str:
    """``$HERMES_HOME/offline/registry.json`` (or ``~/.hermes/...``).

    Kept local to this module rather than importing ``emergency`` so the
    registry has no import edge back into the integration layer.
    """
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(home, "offline", "registry.json")


def _atomic_write_json(path: str, payload: Any) -> None:
    """Write ``payload`` as JSON to ``path`` atomically (temp + fsync + replace)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1_000_000)}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Never leave a temp turd behind on failure.
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    # Best-effort directory fsync so the rename itself is durable.
    try:
        dfd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


class ModelRegistry:
    """An in-memory view of the on-disk model registry.

    Records are keyed by :attr:`hermes_offline.models.ModelRecord.key`
    (``publisher/name@revision``).  The class is intentionally forgiving: an
    absent file is an empty registry and a corrupt file is quarantined, because
    a damaged registry must never take down ``hermes offline status``.
    """

    def __init__(self, path: Optional[str] = None, *, auto_load: bool = True) -> None:
        self.path = str(path) if path else default_registry_path()
        self._records: "OrderedDict[str, ModelRecord]" = OrderedDict()
        #: Non-empty when the previous ``load()`` had to recover from a problem.
        self.load_error: str = ""
        #: Path the corrupt registry was moved to, if quarantine happened.
        self.quarantined_path: str = ""
        if auto_load:
            self.load()

    # -- persistence -------------------------------------------------------

    def load(self) -> "ModelRegistry":
        """Load (or reload) the registry from disk.

        Returns ``self`` so it can be chained.  Missing file -> empty registry.
        Unparseable file -> quarantined, empty registry, ``load_error`` set.
        """
        self.load_error = ""
        self.quarantined_path = ""
        self._records.clear()
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return self
        except (OSError, ValueError) as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self._quarantine_corrupt()
            return self

        records = data.get("records") if isinstance(data, dict) else data
        if not isinstance(records, list):
            self.load_error = "registry document has no 'records' list"
            self._quarantine_corrupt()
            return self

        for item in records:
            if not isinstance(item, dict):
                continue
            try:
                record = ModelRecord.from_dict(item)
            except Exception:  # noqa: BLE001 - one bad row must not kill the file
                continue
            if not record.name:
                continue
            self._records[record.key] = record
        return self

    def reload(self) -> "ModelRegistry":
        return self.load()

    def save(self) -> None:
        """Persist the registry atomically."""
        payload = {
            "version": REGISTRY_VERSION,
            "generated_at": time.time(),
            "count": len(self._records),
            "records": [r.to_dict() for r in self._records.values()],
        }
        _atomic_write_json(self.path, payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": REGISTRY_VERSION,
            "count": len(self._records),
            "records": [r.to_dict() for r in self._records.values()],
        }

    def _quarantine_corrupt(self) -> None:
        if not os.path.exists(self.path):
            return
        dest = f"{self.path}.corrupt-{int(time.time())}"
        try:
            os.replace(self.path, dest)
            self.quarantined_path = dest
        except OSError:
            self.quarantined_path = ""

    # -- introspection -----------------------------------------------------

    def all(self) -> List[ModelRecord]:
        """Every record, in insertion order."""
        return list(self._records.values())

    def get(self, key: str) -> Optional[ModelRecord]:
        """Look up by :attr:`ModelRecord.key` (falls back to a unique name)."""
        if key in self._records:
            return self._records[key]
        matches = [r for r in self._records.values() if r.name == key]
        if len(matches) == 1:
            return matches[0]
        return None

    def find(self, query: str) -> List[ModelRecord]:
        """Case-insensitive substring match over name/publisher/quantization."""
        q = (query or "").strip().lower()
        if not q:
            return self.all()
        out: List[ModelRecord] = []
        for r in self._records.values():
            hay = " ".join([r.name, r.publisher, r.quantization, r.runtime, r.architecture]).lower()
            if q in hay:
                out.append(r)
        return out

    def by_state(self, *states: ModelState) -> List[ModelRecord]:
        wanted = {s if isinstance(s, ModelState) else ModelState(s) for s in states}
        return [r for r in self._records.values() if r.state in wanted]

    def installed(self) -> List[ModelRecord]:
        return self.by_state(ModelState.INSTALLED, ModelState.VERIFIED)

    def verified(self) -> List[ModelRecord]:
        return self.by_state(ModelState.VERIFIED)

    def pinned(self) -> List[ModelRecord]:
        return [r for r in self._records.values() if r.pinned]

    def total_disk_usage_gb(self) -> float:
        return round(sum(float(r.disk_usage_gb or 0.0) for r in self._records.values()), 3)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ModelRecord]:
        return iter(self._records.values())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._records

    # -- mutation ----------------------------------------------------------

    def add(self, record: ModelRecord, *, replace: bool = False) -> ModelRecord:
        """Insert ``record``.

        Raises :class:`ValueError` when the key already exists and
        ``replace`` is false — callers that want upsert semantics should use
        :meth:`add_discovered` / :meth:`update`.
        """
        if not isinstance(record, ModelRecord):
            raise TypeError("record must be a ModelRecord")
        if not record.name:
            raise ValueError("ModelRecord.name is required")
        key = record.key
        if key in self._records and not replace:
            raise ValueError(f"model already registered: {key!r}")
        if record.discovered_at is None:
            record.discovered_at = time.time()
        self._records[key] = record
        return record

    def add_discovered(self, record: ModelRecord, *, replace: bool = True) -> ModelRecord:
        """Upsert a discovered/imported record.

        The record's *own* state is preserved — discovery is how a record
        enters the registry, but an import pipeline may add one that is already
        ``installed``/``verified``.  Forcing ``DISCOVERED`` here would erase
        real lifecycle state.
        """
        if not isinstance(record, ModelRecord):
            raise TypeError("record must be a ModelRecord")
        if not record.name:
            raise ValueError("ModelRecord.name is required")
        if not replace and record.key in self._records:
            raise ValueError(f"model already registered: {record.key!r}")
        previous = self._records.get(record.key)
        if previous is not None:
            # Preserve first-seen discovery time across upserts.
            record.discovered_at = record.discovered_at or previous.discovered_at
        if record.discovered_at is None:
            record.discovered_at = time.time()
        record.last_checked = time.time()
        self._records[record.key] = record
        return record

    def update(self, record: ModelRecord) -> ModelRecord:
        return self.add_discovered(record, replace=True)

    def remove(self, key: str, *, allow_pinned: bool = False) -> bool:
        """Delete a record.  Pinned records are protected unless forced."""
        record = self.get(key)
        if record is None:
            return False
        if record.pinned and not allow_pinned:
            return False
        del self._records[record.key]
        return True

    def clear(self) -> None:
        self._records.clear()

    # -- lifecycle helpers -------------------------------------------------

    def _require(self, key: str) -> ModelRecord:
        record = self.get(key)
        if record is None:
            raise KeyError(key)
        return record

    def mark_downloading(self, key: str, *, source_url: str = "") -> ModelRecord:
        rec = self._require(key)
        rec.state = ModelState.DOWNLOADING
        if source_url:
            rec.source_url = source_url
        rec.last_checked = time.time()
        return rec

    def mark_installed(
        self,
        key: str,
        *,
        local_path: str = "",
        sha256: str = "",
        disk_usage_gb: Optional[float] = None,
    ) -> ModelRecord:
        rec = self._require(key)
        if local_path:
            rec.local_path = local_path
        if sha256:
            rec.sha256 = sha256
        if disk_usage_gb is None and rec.local_path and os.path.exists(rec.local_path):
            try:
                disk_usage_gb = os.path.getsize(rec.local_path) / (1024 ** 3)
            except OSError:
                disk_usage_gb = rec.disk_usage_gb
        if disk_usage_gb is not None:
            rec.disk_usage_gb = float(disk_usage_gb)
        rec.state = ModelState.INSTALLED
        rec.installed_at = rec.installed_at or time.time()
        rec.last_checked = time.time()
        rec.last_error = ""
        return rec

    def mark_verified(
        self,
        key: str,
        *,
        sha256: str = "",
        local_path: str = "",
    ) -> ModelRecord:
        rec = self._require(key)
        if sha256:
            rec.sha256 = sha256
        if local_path:
            rec.local_path = local_path
        if not rec.local_path:
            raise ValueError("cannot verify a model without a local_path")
        rec.state = ModelState.VERIFIED
        rec.last_verified = time.time()
        rec.last_checked = time.time()
        rec.fail_count = 0
        rec.last_error = ""
        return rec

    def mark_failed(self, key: str, error: str = "") -> ModelRecord:
        rec = self._require(key)
        rec.state = ModelState.FAILED
        rec.fail_count += 1
        rec.last_error = error
        rec.last_checked = time.time()
        return rec

    def touch(self, key: str) -> ModelRecord:
        rec = self._require(key)
        rec.last_checked = time.time()
        return rec

    def set_channel(self, key: str, channel: Channel) -> ModelRecord:
        rec = self._require(key)
        rec.channel = channel if isinstance(channel, Channel) else Channel(channel)
        return rec

    def pin(self, key: str, *, pinned: bool = True) -> ModelRecord:
        rec = self._require(key)
        rec.pinned = bool(pinned)
        return rec

    # -- maintenance -------------------------------------------------------

    def prune(
        self,
        *,
        keep_verified: bool = True,
        keep_pinned: bool = True,
        remove_states: Iterable[ModelState] = (ModelState.DISCOVERED,),
        max_records: Optional[int] = None,
        dry_run: bool = False,
    ) -> List[ModelRecord]:
        """Drop low-value records.

        Safety rules (in order): pinned and verified records are never removed
        when the corresponding flag is set; otherwise only records in
        ``remove_states`` are candidates.  ``max_records`` additionally trims
        the oldest non-protected records.  Returns the removed records.
        """
        remove_set = {s if isinstance(s, ModelState) else ModelState(s) for s in remove_states}
        removed: List[ModelRecord] = []

        def protected(rec: ModelRecord) -> bool:
            return bool(
                (keep_verified and rec.state == ModelState.VERIFIED)
                or (keep_pinned and rec.pinned)
            )

        for rec in list(self._records.values()):
            if protected(rec):
                continue
            if rec.state in remove_set:
                removed.append(rec)
        if max_records is not None:
            remaining = [r for r in self._records.values() if r not in removed]
            if len(remaining) > max_records:
                # Oldest (by discovery time) first.
                candidates = sorted(
                    (r for r in remaining if not protected(r)),
                    key=lambda r: r.discovered_at or 0.0,
                )
                excess = len(remaining) - max_records
                removed.extend(candidates[:excess])

        if not dry_run:
            for rec in removed:
                self._records.pop(rec.key, None)
        return removed

    # -- import / export ---------------------------------------------------

    @classmethod
    def from_records(cls, records: Iterable[ModelRecord], path: Optional[str] = None) -> "ModelRegistry":
        reg = cls(path, auto_load=False)
        for rec in records:
            reg.add_discovered(rec, replace=True)
        return reg

    def export_records(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._records.values()]


__all__ = ["ModelRegistry", "REGISTRY_VERSION", "default_registry_path"]
