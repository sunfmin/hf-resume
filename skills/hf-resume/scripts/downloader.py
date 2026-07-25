# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub==1.16.4"]
# ///
"""hf-resume downloader — resumable Hugging Face repo download with a JSON progress feed.

Why the pin: huggingface_hub >= ~1.17 writes each attempt to `<blob>.<random8>.incomplete`
and unlinks it in a `finally`, so nothing is ever reusable across runs. 1.16.4 still opens a
deterministic `<etag>.incomplete` in append mode and passes `resume_size` to `http_get`, which
sends a `Range` header. That is the whole reason this script exists.

Why Xet is force-disabled: even in 1.16.4, `resume_size` is passed to `http_get` but NEVER to
`xet_get`, while the incomplete file is opened `"ab"`. On a Xet-backed repo a leftover partial
would get a full reconstruction appended onto it — silent corruption, not resume.

Progress is derived by polling the filesystem, not from tqdm: tqdm's `desc` is the CDN
basename truncated to 40 chars, and a repo can hold several files with the same basename.
Both cache and local-dir layouts end their temp file with `{etag}.incomplete`, so one rule
maps every file to its bytes-on-disk.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

# --- Environment must be fixed BEFORE huggingface_hub is imported -------------------------
# constants.py snapshots these into module-level values at import time.


def _cli_endpoint() -> str | None:
    """Peek at --endpoint before huggingface_hub imports (constants.py snapshots env at import)."""
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--endpoint" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--endpoint="):
            return a.split("=", 1)[1]
    return None


def _prepare_env() -> list[str]:
    notes: list[str] = []

    # Resume exists only on the plain-HTTP path. Non-negotiable.
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    # hf_transfer has no resume at all.
    if os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None):
        notes.append("unset HF_HUB_ENABLE_HF_TRANSFER (no resume support)")

    # 10s (the default) snaps constantly on a long-haul link; each snap costs a reconnect.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    # An explicitly requested endpoint always wins — a working mirror is a real thing to want.
    forced = _cli_endpoint()
    if forced:
        os.environ["HF_ENDPOINT"] = forced
        notes.append(f"endpoint forced to {forced}")
        return notes

    # Otherwise: mirrors that redirect are worse than nothing. hf-mirror.com now 308-redirects
    # every resolve to huggingface.co (no acceleration) and breaks metadata resolution outright.
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint and "huggingface.co" not in endpoint:
        if os.environ.get("HF_RESUME_KEEP_ENDPOINT") == "1":
            notes.append(f"keeping HF_ENDPOINT={endpoint} (HF_RESUME_KEEP_ENDPOINT=1)")
        else:
            os.environ.pop("HF_ENDPOINT")
            notes.append(
                f"ignored HF_ENDPOINT={endpoint} — it redirects and breaks metadata; "
                "force with --endpoint URL"
            )
    return notes


def _detach() -> None:
    """Leave the parent's process group so a group-wide kill cannot take us down.

    `nohup` only ignores SIGHUP. When the launching terminal/session/agent is killed, the
    whole process group gets signalled and the download dies with it — which defeats the
    entire point of running detached. os.setsid() makes this process a session leader, so
    only an explicit kill of THIS pid stops it (which is what `hf-resume stop` does).
    """
    try:
        os.setsid()
    except OSError:
        pass  # already a session leader, or not permitted — harmless either way


_detach()
ENV_NOTES = _prepare_env()

from huggingface_hub import HfApi, snapshot_download  # noqa: E402
from huggingface_hub.utils import HfHubHTTPError  # noqa: E402

try:  # identical include/exclude semantics to snapshot_download
    from huggingface_hub.utils import filter_repo_objects
except ImportError:  # pragma: no cover
    filter_repo_objects = None

# Retrying these is pointless — they are verdicts about the repo, not about the network.
try:
    from huggingface_hub.errors import (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    FATAL_ERRORS: tuple[type[BaseException], ...] = (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
except ImportError:  # pragma: no cover
    FATAL_ERRORS = ()


# --- State file ---------------------------------------------------------------------------


class State:
    """Atomically-published progress snapshot. The Swift window is a pure reader of this."""

    def __init__(self, path: Path, repo_id: str, repo_type: str, dest: str):
        self.path = path
        self.lock = threading.Lock()
        self.samples: deque[tuple[float, int]] = deque(maxlen=40)
        self.file_samples: dict[str, deque[tuple[float, int]]] = {}
        self.data: dict = {
            "version": 1,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "dest": dest,
            "status": "starting",
            "error": None,
            "notes": ENV_NOTES,
            "pid": os.getpid(),
            "attempt": 1,
            "started_at": time.time(),
            "updated_at": time.time(),
            "total_bytes": 0,
            "done_bytes": 0,
            "resumed_bytes": 0,
            "speed_bps": 0.0,
            "eta_seconds": None,
            "files": [],
        }

    def publish(self) -> None:
        with self.lock:
            self.data["updated_at"] = time.time()
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data))
            os.replace(tmp, self.path)  # atomic: readers never see a half-written file

    def set(self, **kw) -> None:
        with self.lock:
            self.data.update(kw)

    def rate(self, done: int) -> float:
        """Overall bytes/s over a ~6s trailing window."""
        now = time.time()
        self.samples.append((now, done))
        while len(self.samples) > 2 and now - self.samples[0][0] > 6.0:
            self.samples.popleft()
        if len(self.samples) < 2:
            return 0.0
        (t0, b0), (t1, b1) = self.samples[0], self.samples[-1]
        return (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0

    def file_rate(self, key: str, done: int) -> float:
        dq = self.file_samples.setdefault(key, deque(maxlen=20))
        now = time.time()
        dq.append((now, done))
        while len(dq) > 2 and now - dq[0][0] > 5.0:
            dq.popleft()
        if len(dq) < 2:
            return 0.0
        (t0, b0), (t1, b1) = dq[0], dq[-1]
        return (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0


# --- Repo introspection -------------------------------------------------------------------


def _lfs_sha(lfs) -> str | None:
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256") or lfs.get("oid")
    return getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)


def list_files(api: HfApi, repo_id: str, repo_type: str, revision: str | None):
    """-> (commit_sha, [{path, size, etag}]). One HTTP call via files_metadata."""
    info = api.repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True)
    out = []
    for sib in info.siblings or []:
        etag = _lfs_sha(getattr(sib, "lfs", None)) or getattr(sib, "blob_id", None)
        out.append(
            {
                "path": sib.rfilename,
                "size": int(getattr(sib, "size", 0) or 0),
                "etag": etag,
            }
        )
    return info.sha, out


def apply_patterns(files, include, exclude):
    paths = [f["path"] for f in files]
    if filter_repo_objects is not None:
        kept = set(
            filter_repo_objects(paths, allow_patterns=include or None, ignore_patterns=exclude or None)
        )
    else:  # pragma: no cover - fallback mirrors fnmatch semantics
        kept = set(paths)
        if include:
            kept = {p for p in kept if any(fnmatch.fnmatch(p, g) for g in include)}
        if exclude:
            kept = {p for p in kept if not any(fnmatch.fnmatch(p, g) for g in exclude)}
    return [f for f in files if f["path"] in kept]


# --- Bytes-on-disk resolution -------------------------------------------------------------


def repo_folder(repo_id: str, repo_type: str) -> str:
    prefix = {"model": "models", "dataset": "datasets", "space": "spaces"}[repo_type]
    return f"{prefix}--" + repo_id.replace("/", "--")


class Locator:
    """Maps each repo file to (final_path, incomplete_globs) for both cache and local-dir."""

    def __init__(self, repo_id, repo_type, revision_sha, local_dir: str | None, cache_dir: str | None):
        self.local_dir = Path(local_dir).expanduser().resolve() if local_dir else None
        if self.local_dir is None:
            from huggingface_hub import constants

            root = Path(cache_dir or constants.HF_HUB_CACHE)
            self.repo_root = root / repo_folder(repo_id, repo_type)
            self.blobs = self.repo_root / "blobs"
            self.snapshot = self.repo_root / "snapshots" / revision_sha
        else:
            self.incomplete_root = self.local_dir / ".cache" / "huggingface" / "download"

    def final_path(self, rel: str) -> Path:
        if self.local_dir is not None:
            return self.local_dir / rel
        return self.snapshot / rel

    def blob_path(self, etag: str | None) -> Path | None:
        if self.local_dir is not None or not etag:
            return None
        return self.blobs / etag

    def incomplete_bytes(self, rel: str, etag: str | None) -> int:
        """Bytes already written to the temp file. Both layouts end with '{etag}.incomplete'."""
        if not etag:
            return 0
        if self.local_dir is None:
            p = self.blobs / f"{etag}.incomplete"
            try:
                return p.stat().st_size
            except OSError:
                return 0
        # local-dir: <shorthash>.<etag>.incomplete next to the mirrored relative path
        parent = (self.incomplete_root / rel).parent
        try:
            for p in parent.glob(f"*{etag}.incomplete"):
                return p.stat().st_size
        except OSError:
            pass
        return 0

    def done_bytes(self, rel: str, etag: str | None, size: int) -> int | None:
        """Returns size if the file is fully materialised, else None."""
        blob = self.blob_path(etag)
        if blob is not None:
            try:
                st = blob.stat()
                if size == 0 or st.st_size == size:
                    return st.st_size
            except OSError:
                pass
        fp = self.final_path(rel)
        try:
            st = fp.stat()  # follows the snapshot symlink
            if size == 0 or st.st_size == size:
                return st.st_size
        except OSError:
            pass
        return None


# --- Monitor ------------------------------------------------------------------------------


def monitor(state: State, loc: Locator, files, stop: threading.Event, interval: float = 0.35):
    while not stop.is_set():
        total_done = 0
        rows = []
        for f in files:
            rel, size, etag = f["path"], f["size"], f["etag"]
            complete = loc.done_bytes(rel, etag, size)
            if complete is not None:
                done, st = complete, "done"
            else:
                done = loc.incomplete_bytes(rel, etag)
                st = "active" if done > 0 else "queued"
            total_done += done
            rows.append(
                {
                    "path": rel,
                    "size": size,
                    "done": done,
                    "state": st,
                    "speed_bps": state.file_rate(rel, done) if st == "active" else 0.0,
                    "resumed_from": f.get("resumed_from", 0),
                }
            )

        speed = state.rate(total_done)
        remaining = max(0, state.data["total_bytes"] - total_done)
        state.set(
            done_bytes=total_done,
            files=rows,
            speed_bps=speed,
            eta_seconds=(remaining / speed) if speed > 1024 else None,
        )
        state.publish()
        stop.wait(interval)


# --- Main ---------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(prog="hf-resume-downloader")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--type", default="model", choices=["model", "dataset", "space"])
    ap.add_argument("--revision", default=None)
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--endpoint",
        default=None,
        help="force an HF endpoint/mirror (already applied at import; listed here for --help)",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tries", type=int, default=200)
    ap.add_argument("--state", required=True)
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="huggingface_hub debug logs, incl. its own '(resume from N/M)' line",
    )
    args = ap.parse_args()

    if args.verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)

    state_path = Path(args.state).expanduser()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    dest = args.local_dir or (args.cache_dir or "")
    state = State(state_path, args.repo, args.type, dest)
    state.set(status="resolving")
    state.publish()
    for n in ENV_NOTES:
        print(f"[env] {n}", flush=True)

    try:
        sha, all_files = list_files(api, args.repo, args.type, args.revision)
    except Exception as e:  # noqa: BLE001
        state.set(status="error", error=f"could not list repo: {e}")
        state.publish()
        print(f"[fatal] {e}", file=sys.stderr, flush=True)
        return 2

    files = apply_patterns(all_files, args.include, args.exclude)
    if not files:
        state.set(status="error", error="no files matched --include/--exclude")
        state.publish()
        return 2

    loc = Locator(args.repo, args.type, sha, args.local_dir, args.cache_dir)
    if not loc.local_dir:
        state.set(dest=str(loc.snapshot))

    # Record what we start from, so the window can show "resumed" honestly.
    resumed = 0
    for f in files:
        if loc.done_bytes(f["path"], f["etag"], f["size"]) is None:
            f["resumed_from"] = loc.incomplete_bytes(f["path"], f["etag"])
            resumed += f["resumed_from"]

    total = sum(f["size"] for f in files)
    state.set(status="running", total_bytes=total, revision=sha, resumed_bytes=resumed)
    state.publish()
    print(
        f"[start] {args.repo} · {len(files)} files · {total / 1e9:.2f} GB"
        + (f" · resuming from {resumed / 1e6:.1f} MB on disk" if resumed else ""),
        flush=True,
    )
    # Per-file proof of resume, so the log answers "did it really continue?" without --verbose.
    for f in files:
        if f.get("resumed_from"):
            print(f"[resume from {f['resumed_from']}/{f['size']}] {f['path']}", flush=True)

    stop = threading.Event()
    mon = threading.Thread(target=monitor, args=(state, loc, files, stop), daemon=True)
    mon.start()

    def _term(signum, _frame):
        state.set(status="stopped", error=None)
        stop.set()
        state.publish()
        print(f"[stopped] signal {signum}; partials kept — rerun to resume", flush=True)
        os._exit(0)

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    kwargs = dict(
        repo_id=args.repo,
        repo_type=args.type,
        revision=args.revision,
        allow_patterns=args.include or None,
        ignore_patterns=args.exclude or None,
        max_workers=max(1, args.workers),
    )
    if args.local_dir:
        kwargs["local_dir"] = args.local_dir
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir

    def bytes_on_disk() -> int:
        return sum(
            (loc.done_bytes(f["path"], f["etag"], f["size"]) or loc.incomplete_bytes(f["path"], f["etag"]))
            for f in files
        )

    # Resume makes retrying nearly free: every attempt picks up where the last died. But an
    # attempt that fails WITHOUT moving any bytes is not a flaky link — it is a broken endpoint,
    # a bad token, or a vanished repo. Retrying that 200 times just spins for an hour, so bail
    # after STALL_LIMIT consecutive zero-progress failures.
    STALL_LIMIT = 5
    delay, stalled = 3.0, 0
    for attempt in range(1, args.max_tries + 1):
        state.set(attempt=attempt, status="running", error=None)
        before = bytes_on_disk()
        try:
            snapshot_download(**kwargs)
            break
        except KeyboardInterrupt:
            _term(signal.SIGINT, None)
        except FATAL_ERRORS as e:
            msg = f"{type(e).__name__}: {e}"[:400]
            print(f"[fatal] {msg}", flush=True)
            state.set(status="error", error=msg)
            stop.set()
            state.publish()
            return 2
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"[:400]
            progressed = bytes_on_disk() - before
            stalled = 0 if progressed > 0 else stalled + 1
            print(
                f"[retry {attempt}/{args.max_tries}] +{progressed}B"
                + (f" · stalled {stalled}/{STALL_LIMIT}" if stalled else "")
                + f" · {msg}",
                flush=True,
            )
            state.set(status="retrying", error=msg)
            state.publish()
            if stalled >= STALL_LIMIT or attempt >= args.max_tries:
                reason = (
                    f"{STALL_LIMIT} attempts in a row downloaded nothing — check the endpoint, "
                    f"token and repo id. Last error: {msg}"
                    if stalled >= STALL_LIMIT
                    else msg
                )
                state.set(status="error", error=reason)
                stop.set()
                state.publish()
                print(f"[error] {reason}", flush=True)
                return 1
            time.sleep(min(delay, 30.0))
            delay = 3.0 if progressed > 0 else delay * 1.6

    # Final truth: re-check every file on disk before declaring success.
    stop.set()
    mon.join(timeout=2.0)
    missing = [
        f["path"] for f in files if loc.done_bytes(f["path"], f["etag"], f["size"]) is None
    ]
    total_done = sum(
        (loc.done_bytes(f["path"], f["etag"], f["size"]) or loc.incomplete_bytes(f["path"], f["etag"]))
        for f in files
    )
    state.set(
        done_bytes=total_done,
        files=[
            {
                "path": f["path"],
                "size": f["size"],
                "done": loc.done_bytes(f["path"], f["etag"], f["size"]) or 0,
                "state": "done" if f["path"] not in missing else "queued",
                "speed_bps": 0.0,
                "resumed_from": f.get("resumed_from", 0),
            }
            for f in files
        ],
        speed_bps=0.0,
        eta_seconds=0,
        status="done" if not missing else "error",
        error=None if not missing else f"{len(missing)} file(s) incomplete: {missing[:5]}",
    )
    state.publish()
    print(
        f"[done] {args.repo} → {state.data['dest']}"
        if not missing
        else f"[error] incomplete: {missing[:5]}",
        flush=True,
    )
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
