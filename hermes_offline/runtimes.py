"""Local inference runtime management for offline operation.

The offline subsystem never embeds an inference engine itself.  Instead it
detects runtimes the user already has (or deliberately installed) and exposes
them through a thin, uniform adapter so a local model can be presented to
Hermes as an ordinary OpenAI-compatible provider::

    Local model (GGUF/...)  ->  RuntimeAdapter  ->  Hermes provider (OpenAI API)

Only two runtimes are supported on purpose (``ollama`` and ``llama.cpp``);
adding more would be unnecessary surface area.  Detection is passive and
stdlib-only: a binary on ``PATH`` and/or a reachable TCP endpoint.  Nothing is
started or downloaded implicitly -- a runtime the user did not install is
simply reported as unavailable.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import ModelRecord, RuntimeInfo

# Well-known local endpoints.  These are defaults only -- every port/host is
# overridable via configuration so no particular setup is hardcoded.
OLLAMA_DEFAULT_HOST = "127.0.0.1"
OLLAMA_DEFAULT_PORT = 11434
LLAMA_CPP_DEFAULT_HOST = "127.0.0.1"
LLAMA_CPP_DEFAULT_PORT = 8080


def _tcp_reachable(host: str, port: int, timeout_s: float = 0.75) -> bool:
    """True if a TCP connection to ``host:port`` can be established quickly."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def _http_get_json(url: str, timeout_s: float = 2.5) -> Optional[Any]:
    """Bounded, never-raising GET returning parsed JSON or ``None``."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (localhost)
            raw = resp.read(1_000_000)
        return json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return None


class RuntimeAdapter:
    """Uniform interface over a local inference runtime."""

    #: short stable identifier, e.g. ``"ollama"``
    name: str = "runtime"
    #: human label used in doctor / readiness output
    label: str = "Local runtime"
    #: API surface exposed to Hermes (both speak the OpenAI dialect)
    api: str = "openai"

    def __init__(self, *, host: str, port: int, timeout_s: float = 2.5) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    # -- capability probes -------------------------------------------------
    def is_available(self) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def list_models(self) -> List[str]:
        return []

    def openai_base_url(self) -> str:
        """Base URL Hermes should use for this runtime as an OpenAI provider."""
        return f"http://{self.host}:{self.port}/v1"

    def detail(self) -> str:
        return ""

    def info(self) -> RuntimeInfo:
        available = False
        models: List[str] = []
        try:
            available = self.is_available()
            if available:
                models = self.list_models()
        except Exception:  # never let a probe crash readiness/doctor
            available = False
        version = ""
        try:
            version = self._version()
        except Exception:
            version = ""
        return RuntimeInfo(
            name=self.name,
            available=available,
            version=version,
            endpoint=f"http://{self.host}:{self.port}",
            models=models,
            detail=f"{self.api} api; base_url={self.openai_base_url()}; {self.detail()}",
        )

    def _version(self) -> str:
        return ""

    # -- provider integration ---------------------------------------------
    def provider_profile(self, model: str) -> Dict[str, Any]:
        """A description Hermes can use to register this as an OpenAI-compatible
        provider.  ``api_key`` is a non-secret placeholder: local runtimes do not
        authenticate, and we deliberately never invent a credential."""
        return {
            "name": f"local-{self.name}",
            "kind": "openai_compatible",
            "base_url": self.openai_base_url(),
            "api_key": "local-no-auth",
            "model": model,
            "note": "Local runtime; no external network required.",
        }

    def start(self, model: str) -> bool:
        """Best-effort start.  Returns False when we cannot start it ourselves --
        the caller should then report the exact manual command, never pretend."""
        return False

    def stop(self) -> bool:
        return False


class OllamaRuntime(RuntimeAdapter):
    name = "ollama"
    label = "Ollama"

    def _host_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_available(self) -> bool:
        if shutil.which("ollama") is None and not _tcp_reachable(self.host, self.port):
            return False
        return _http_get_json(f"{self._host_root()}/api/tags", self.timeout_s) is not None

    def list_models(self) -> List[str]:
        data = _http_get_json(f"{self._host_root()}/api/tags", self.timeout_s)
        if not isinstance(data, dict):
            return []
        out: List[str] = []
        for entry in data.get("models", []) or []:
            if isinstance(entry, dict) and entry.get("name"):
                out.append(str(entry["name"]))
        return out

    def detail(self) -> str:
        binary = shutil.which("ollama")
        return f"binary={binary or 'not on PATH'}"

    def start(self, model: str) -> bool:
        """If the daemon is up we can ask it to serve a model that is already
        pulled.  We never pull (that needs the network)."""
        if not _tcp_reachable(self.host, self.port):
            binary = shutil.which("ollama")
            if binary is None:
                return False
        return self.is_available()


class LlamaCppRuntime(RuntimeAdapter):
    name = "llama.cpp"
    label = "llama.cpp"

    def _binary(self) -> Optional[str]:
        for candidate in ("llama-server", "llama_server", "server"):
            path = shutil.which(candidate)
            if path:
                return path
        return None

    def is_available(self) -> bool:
        # Either a server binary is present, or something already listens on the port.
        return self._binary() is not None or _tcp_reachable(self.host, self.port)

    def list_models(self) -> List[str]:
        data = _http_get_json(f"http://{self.host}:{self.port}/v1/models", self.timeout_s)
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return [str(m.get("id")) for m in data["data"] if isinstance(m, dict) and m.get("id")]
        return []

    def detail(self) -> str:
        return f"binary={self._binary() or 'not on PATH'}"

    def start(self, model_path: str) -> bool:
        binary = self._binary()
        if binary is None or not os.path.exists(model_path):
            return False
        try:  # detach; the runtime keeps serving in the background
            subprocess.Popen(  # noqa: S603
                [binary, "-m", model_path, "--host", self.host, "--port", str(self.port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except OSError:
            return False


_RUNTIME_NAMES = {
    "ollama": OllamaRuntime,
    "llama.cpp": LlamaCppRuntime,
    "llamacpp": LlamaCppRuntime,
    "llama_server": LlamaCppRuntime,
    "gguf": LlamaCppRuntime,
}


class RuntimeManager:
    """Discovers and selects local runtimes."""

    def __init__(
        self,
        *,
        ollama_host: str = OLLAMA_DEFAULT_HOST,
        ollama_port: int = OLLAMA_DEFAULT_PORT,
        llama_cpp_host: str = LLAMA_CPP_DEFAULT_HOST,
        llama_cpp_port: int = LLAMA_CPP_DEFAULT_PORT,
        timeout_s: float = 2.5,
    ) -> None:
        self.adapters: Dict[str, RuntimeAdapter] = {
            "ollama": OllamaRuntime(host=ollama_host, port=ollama_port, timeout_s=timeout_s),
            "llama.cpp": LlamaCppRuntime(host=llama_cpp_host, port=llama_cpp_port, timeout_s=timeout_s),
        }

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]] = None) -> "RuntimeManager":
        cfg = cfg or {}
        return cls(
            ollama_host=cfg.get("ollama_host", OLLAMA_DEFAULT_HOST),
            ollama_port=int(cfg.get("ollama_port", OLLAMA_DEFAULT_PORT)),
            llama_cpp_host=cfg.get("llama_cpp_host", LLAMA_CPP_DEFAULT_HOST),
            llama_cpp_port=int(cfg.get("llama_cpp_port", LLAMA_CPP_DEFAULT_PORT)),
            timeout_s=float(cfg.get("timeout_s", 2.5)),
        )

    def get(self, name: str) -> Optional[RuntimeAdapter]:
        key = (name or "").strip().lower()
        cls = _RUNTIME_NAMES.get(key)
        if cls is None:
            return None
        return self.adapters.get("ollama" if cls is OllamaRuntime else "llama.cpp")

    def runtime_for(self, record: ModelRecord) -> Optional[RuntimeAdapter]:
        """Pick the adapter a record declares (``record.runtime`` /
        ``record.extra['runtime']``).

        A declared runtime is honoured even when it is not currently running:
        the caller may start it, and silently substituting a different engine
        could load an incompatible artifact.  We only fall back to any
        available runtime when the record does not name one.
        """
        declared = str(getattr(record, "runtime", "") or record.extra.get("runtime", "") or "")
        if declared:
            adapter = self.get(declared)
            if adapter is not None:
                return adapter
        for candidate in self.adapters.values():
            if candidate.is_available():
                return candidate
        return None

    def detect(self) -> List[RuntimeInfo]:
        return [adapter.info() for adapter in self.adapters.values()]

    def available(self) -> List[RuntimeInfo]:
        return [info for info in self.detect() if info.available]
