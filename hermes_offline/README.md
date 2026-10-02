# hermes_offline — resilient & offline operation

`hermes_offline` is the subsystem that keeps Hermes usable when
international connectivity is flaky or entirely gone. It is **offline-first**
and **stdlib-only** (no new runtime dependencies) and integrates with the
existing Hermes CLI, registry, runtime and doctor surfaces rather than
creating a parallel architecture.

## Why it exists

Hermes normally assumes reachable model providers. When DNS, TLS, or the
provider endpoint is blocked, users got an opaque failure. This subsystem:

1. **Classifies** the failure instead of treating everything as "offline".
2. **Bounds** retries with exponential backoff + jitter and per-provider
   circuit breakers, so a blocked provider never spins forever.
3. **Falls back** to a verified *local* model served by a local runtime.
4. **Verifies** every artifact by checksum and never marks a corrupted or
   incomplete download as ready.
5. **Recovers** interrupted long tasks from checkpoints.
6. **Documents** itself offline and reports a real readiness verdict.

## Connectivity states

| State | Meaning |
| --- | --- |
| `FULL_ONLINE` | general + provider endpoints reachable and fast |
| `DEGRADED` | reachable but slow, or only some endpoints respond |
| `PROVIDER_UNAVAILABLE` | general internet works, provider endpoint does not |
| `OFFLINE` | no general endpoint reachable (DNS/TLS/timeout/connrefused) |
| `EMERGENCY_OFFLINE_READY` | `OFFLINE` **and** a verified local fallback is ready |

Failure classes reported per probe: `DNS`, `TLS`, `TIMEOUT`, `CONNECTION`,
`RATE_LIMITED`, `HTTP_4XX`, `HTTP_5XX`, `UNKNOWN`.

Unified proxies are supported but never hardcoded — configure via
`ConnectivityConfig` / environment only.

## CLI

```bash
hermes offline status            # readiness + connectivity + emergency mode
hermes offline status --json     # machine readable
hermes offline models            # registry: installed / verified
hermes offline prepare --dry-run # show the plan without downloading
hermes offline prepare --import /path/to/model.gguf
hermes offline verify            # re-check checksums of installed artifacts
hermes offline import FILE --sha256 HASH --name NAME
hermes offline docs              # (re)generate offline documentation
hermes offline recover --task ID # inspect a task checkpoint
hermes doctor --offline          # offline readiness report via doctor
```

## Local model & runtime management

* A **model registry** (`$HERMES_HOME/offline/registry.json`) records every
  artifact: source, version/revision, license, size, quantization, checksum,
  capabilities, runtime, install/verification timestamps and state
  (`discovered → downloading → installed → verified | failed`).
* A **resumable downloader** writes to `<file>.part`, uses HTTP `Range` to
  resume, verifies SHA-256, and only then atomically renames into place. It
  preflights free disk space and quarantines checksum mismatches.
* **Runtimes** are detected, not bundled: Ollama and llama.cpp/`llama-server`
  are probed and exposed through an OpenAI-compatible base URL so they can be
  used as a normal Hermes provider.

## Offline transfer (air-gapped machines)

1. On a connected machine: `hermes offline import FILE --sha256 HASH`.
2. Move the artifact (USB etc.) to the target machine.
3. On the target: `hermes offline import FILE --sha256 HASH` then
   `hermes offline verify`.

Checksums are always computed locally; a record is only `VERIFIED` when an
expected hash was supplied **and** matches.

## Automatic local provider fallback

When international providers are unreachable, Hermes can fall back to a **local**
model instead of failing. This reuses the existing fallback chain — no parallel
mechanism — by appending one final entry targeting the local runtime's
OpenAI-compatible endpoint.

Enable it in `config.yaml`:

```yaml
offline:
  auto_local_fallback: true
```

Preconditions (all real, never assumed):

1. at least one model in the local registry with state `verified` (or `installed`), and
2. a detected runtime that is actually available (Ollama / llama.cpp).

When both hold, `hermes fallback list` shows the local entry last; it is reached
only after every remote provider has failed. If either precondition is missing,
nothing is appended and the chain is unchanged.

## Security

* No token, key or connection string is ever logged, committed or written to
  docs. Secrets are read from environment/config and redacted in output.
* Downloaded artifacts are checksum-verified before use; unverified artifacts
  are never reported as ready.

## Limitations (honest)

* Discovery is **offline-first**: it reads a bundled/permitted local catalog
  and local imports. There is no automatic scraping of model hubs.
* No inference engine is shipped; a runtime must already be installed.
* Checksums are only as trustworthy as the value you supply for artifacts
  imported from outside.
