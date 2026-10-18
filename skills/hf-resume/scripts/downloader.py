# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub==1.16.4", "socksio"]  # socksio: httpx needs it when all_proxy=socks5://
# ///
"""hf-resume downloader — resumable HF/ModelScope repo download with a JSON progress feed.

Why the pin: huggingface_hub >= ~1.17 writes each attempt to `<blob>.<random8>.incomplete`
and unlinks it in a `finally`, so nothing is ever reusable across runs. 1.16.4 still opens a
deterministic `<etag>.incomplete` in append mode and passes `resume_size` to `http_get`, which
sends a `Range` header. That is the whole reason this script exists.

Why Xet is force-disabled: even in 1.16.4, `resume_size` is passed to `http_get` but NEVER to
`xet_get`, while the incomplete file is opened `"ab"`. On a Xet-backed repo a leftover partial
would get a full reconstruction appended onto it — silent corruption, not resume.

Why ModelScope can share the HF cache: for LFS files the HF cache names each blob by its
sha256, and ModelScope publishes the same sha256 for the same repo (verified byte-identical).
So ModelScope bytes are valid HF cache blobs, and either source populates one interchangeable
cache — switching sources never re-downloads what the other already fetched.

Progress is derived by polling the filesystem, not from tqdm: tqdm's `desc` is the CDN
basename truncated to 40 chars, and a repo can hold several files with the same basename.
Both cache and local-dir layouts end their temp file with `{etag}.incomplete`, so one rule
maps every file to its bytes-on-disk — and the ModelScope backend uses the same names.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

MODELSCOPE_API = "https://www.modelscope.cn/api/v1/models"
# Below this, direct HF is bad enough that the mirror is worth probing. 2 MB/s is ~2.5 h for a
# 17 GB model; ModelScope measured ~3 MB/s from CN networks, so this is roughly the break-even.
HF_PROBE_MIN_MBPS = 2.0

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


_detach()
ENV_NOTES = _prepare_env()

import httpx  # noqa: E402  (comes with huggingface_hub 1.16.4)
from huggingface_hub import HfApi, constants, snapshot_download  # noqa: E402

# Progress here is the size of the `.incomplete` file, and http_get only writes once a full
# chunk arrives. At the 10 MB default the file grows in 10 MB jumps — on a 1 MB/s link that is
# one update every 10 s, so the UI reads 0 B/s and "ETA —" most of the time. 1 MB chunks make
# progress smooth at negligible cost (http_get reads constants at call time, so this is live).
constants.DOWNLOAD_CHUNK_SIZE = 1024 * 1024

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
        # ~20s of 0.35s ticks. Wide enough that the window still reads a real rate when writes
        # land in bursts rather than smoothly.
        self.samples: deque[tuple[float, int]] = deque(maxlen=120)
        self.file_samples: dict[str, deque[tuple[float, int]]] = {}
        self.data: dict = {
            "version": 1,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "source": None,
            "dest": dest,
            "status": "starting",
            "error": None,
            "notes": list(ENV_NOTES),
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

    def note(self, msg: str) -> None:
        with self.lock:
            self.data["notes"] = list(self.data.get("notes") or []) + [msg]
        print(f"[env] {msg}", flush=True)

    def rate(self, done: int) -> float:
        """Overall bytes/s over a ~6s trailing window."""
        now = time.time()
        self.samples.append((now, done))
        while len(self.samples) > 2 and now - self.samples[0][0] > 20.0:
            self.samples.popleft()
        if len(self.samples) < 2:
            return 0.0
        (t0, b0), (t1, b1) = self.samples[0], self.samples[-1]
        return (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0

    def file_rate(self, key: str, done: int) -> float:
        dq = self.file_samples.setdefault(key, deque(maxlen=80))
        now = time.time()
        dq.append((now, done))
        while len(dq) > 2 and now - dq[0][0] > 15.0:
            dq.popleft()
        if len(dq) < 2:
            return 0.0
        (t0, b0), (t1, b1) = dq[0], dq[-1]
        return (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0


# --- Repo introspection: Hugging Face -----------------------------------------------------


def _lfs_sha(lfs) -> str | None:
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256") or lfs.get("oid")
    return getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)


def hf_list_files(api: HfApi, repo_id: str, repo_type: str, revision: str | None):
    """-> (commit_sha, [{path, size, etag, lfs}]). One HTTP call via files_metadata."""
    info = api.repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True)
    out = []
    for sib in info.siblings or []:
        sha = _lfs_sha(getattr(sib, "lfs", None))
        out.append(
            {
                "path": sib.rfilename,
                "size": int(getattr(sib, "size", 0) or 0),
                "etag": sha or getattr(sib, "blob_id", None),
                "lfs": bool(sha),
            }
        )
    return info.sha, out


# --- Repo introspection: ModelScope -------------------------------------------------------


def ms_list_files(repo_id: str, revision: str = "master"):
    """-> (commit_sha, download_rev, [{path, size, etag, lfs}]).

    ModelScope publishes the same sha256 HF uses to name LFS blobs, so `etag` lands the bytes
    in exactly the place the HF backend would have put them.

    Two revisions, deliberately: the file API reports a per-file COMMIT SHA, but the repo
    download endpoint 404s on a commit sha and only accepts a branch name. So the sha names the
    snapshot directory while `download_rev` (the branch) goes into download URLs.
    """
    url = f"{MODELSCOPE_API}/{repo_id}/repo/files"
    with httpx.Client(timeout=30.0, follow_redirects=True) as c:
        r = c.get(url, params={"Revision": revision, "Recursive": "True"})
        r.raise_for_status()
        payload = r.json()
    if payload.get("Code") not in (200, None):
        raise RuntimeError(f"ModelScope listing failed: {str(payload.get('Message'))[:200]}")
    entries = (payload.get("Data") or {}).get("Files") or []
    files, commit_sha = [], None
    for e in entries:
        if e.get("Type") == "tree":
            continue
        path = e.get("Path")
        if not path:
            continue
        commit_sha = commit_sha or e.get("Revision")
        files.append(
            {
                "path": path,
                "size": int(e.get("Size") or 0),
                "etag": e.get("Sha256") if e.get("IsLFS") else None,
                "lfs": bool(e.get("IsLFS")),
                "ms_sha256": e.get("Sha256"),
            }
        )
    return (commit_sha or revision), revision, files


def ms_file_url(repo_id: str, path: str, revision: str) -> str:
    return (
        f"{MODELSCOPE_API}/{repo_id}/repo"
        f"?Revision={quote(revision, safe='')}&FilePath={quote(path, safe='')}"
    )


def git_blob_sha1(path: Path) -> str:
    """HF names non-LFS blobs by their git blob id, so compute it to keep the cache reusable."""
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(b"blob %d\0" % size)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


# --- Source selection ---------------------------------------------------------------------


def detect_proxy() -> list[str]:
    """Report proxy signals — explanation only, never the decision.

    Clash in TUN mode routes at the network layer: `scutil --proxy` shows HTTPEnable: 0 and no
    proxy env vars exist, yet huggingface.co is fast. So proxy state cannot be inferred from
    configuration; it can only be measured. This function exists to explain a measurement.
    """
    signals = []
    for var in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        if os.environ.get(var):
            signals.append(f"{var} set")
            break
    try:
        out = subprocess.run(["scutil", "--proxy"], capture_output=True, text=True, timeout=5).stdout
        if "HTTPSEnable : 1" in out or "HTTPEnable : 1" in out:
            signals.append("system proxy on")
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(
            ["pgrep", "-lf", "-i", "clash|mihomo|sing-box|singbox|surge"],
            capture_output=True, text=True, timeout=5,
        ).stdout.lower()
        for name in ("mihomo", "clash", "sing-box", "singbox", "surge"):
            if name in out:
                signals.append(f"{name} running")
                break
    except Exception:  # noqa: BLE001
        pass
    return signals


def probe_mbps(
    url: str, nbytes: int = 8_000_000, warmup: int = 1_500_000, timeout: float = 20.0
) -> float | None:
    """Steady-state throughput for a short ranged read, or None if unusable.

    The clock starts only after `warmup` bytes have landed: TCP slow start makes a naive
    3 MB timing read ~1 MB/s on a link that actually sustains 8 MB/s, which would send the
    source decision the wrong way.
    """
    try:
        t_wall = time.time()
        got = 0
        t_start: float | None = None
        base = 0
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            with c.stream("GET", url, headers={"Range": f"bytes=0-{nbytes - 1}"}) as r:
                if r.status_code >= 400:
                    return None
                for chunk in r.iter_bytes(1 << 16):
                    got += len(chunk)
                    if t_start is None and got >= warmup:
                        t_start, base = time.time(), got
                    if got >= nbytes or time.time() - t_wall > timeout:
                        break
        if t_start is not None:
            dt, moved = time.time() - t_start, got - base
            if dt > 0.05 and moved > 200_000:
                return (moved / dt) / 1e6
        # Never got past warm-up: fall back to whole-transfer rate (slow links land here).
        dt = time.time() - t_wall
        if got < 100_000 or dt <= 0:
            return None
        return (got / dt) / 1e6
    except Exception:  # noqa: BLE001
        return None


def pick_source(repo_id: str, repo_type: str, requested: str, state: State) -> str:
    """'hf' or 'modelscope'. `auto` measures both and takes the faster reachable one."""
    if requested in ("hf", "modelscope"):
        state.note(f"source forced to {requested}")
        return requested

    signals = detect_proxy()
    sig = ", ".join(signals) if signals else "no proxy signals"

    if repo_type != "model":
        state.note(f"auto → hf (ModelScope mirrors models only; {sig})")
        return "hf"

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    hf_probe = f"{endpoint}/{repo_id}/resolve/main/config.json"
    hf_mbps = probe_mbps(hf_probe)
    if hf_mbps is None:  # not every repo has config.json — retry on the real file list
        try:
            _, files = hf_list_files(HfApi(), repo_id, repo_type, None)
            biggest = max(files, key=lambda f: f["size"], default=None)
            if biggest:
                hf_mbps = probe_mbps(f"{endpoint}/{repo_id}/resolve/main/{quote(biggest['path'])}")
        except Exception:  # noqa: BLE001
            hf_mbps = None

    # Direct is the preference whenever it is usable. Only when HF is unreachable or crawling
    # do we spend time probing the mirror — that is the no-proxy case the mirror exists for.
    # Short probes are noisy in both directions (TCP slow start undershoots, a warm CDN edge
    # overshoots), so they decide only "usable vs not", never a close race.
    if hf_mbps is not None and hf_mbps >= HF_PROBE_MIN_MBPS:
        state.note(f"auto → hf (direct, probe {hf_mbps:.1f} MB/s; {sig})")
        return "hf"

    hf_desc = "unreachable" if hf_mbps is None else f"{hf_mbps:.2f} MB/s"
    ms_mbps = None
    try:
        _sha, dl_rev, msf = ms_list_files(repo_id)
        biggest = max((f for f in msf if f["size"] > 1_000_000), key=lambda f: f["size"], default=None)
        if biggest:
            ms_mbps = probe_mbps(ms_file_url(repo_id, biggest["path"], dl_rev))
    except Exception:  # noqa: BLE001
        ms_mbps = None

    if ms_mbps is None:
        state.note(f"auto → hf (hf {hf_desc} but modelscope unusable too; {sig})")
        return "hf"
    state.note(f"auto → modelscope ({ms_mbps:.1f} MB/s vs hf {hf_desc}; {sig})")
    return "modelscope"


# --- Bytes-on-disk resolution -------------------------------------------------------------


def repo_folder(repo_id: str, repo_type: str) -> str:
    prefix = {"model": "models", "dataset": "datasets", "space": "spaces"}[repo_type]
    return f"{prefix}--" + repo_id.replace("/", "--")


class Locator:
    """Maps each repo file to (final_path, incomplete) for both cache and local-dir."""

    def __init__(self, repo_id, repo_type, revision_sha, local_dir: str | None, cache_dir: str | None):
        self.local_dir = Path(local_dir).expanduser().resolve() if local_dir else None
        if self.local_dir is None:
            from huggingface_hub import constants

            root = Path(cache_dir or constants.HF_HUB_CACHE)
            self.repo_root = root / repo_folder(repo_id, repo_type)
            self.blobs = self.repo_root / "blobs"
            self.snapshot = self.repo_root / "snapshots" / revision_sha
            self.refs = self.repo_root / "refs"
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

    def incomplete_path(self, rel: str, etag: str) -> Path:
        if self.local_dir is None:
            return self.blobs / f"{etag}.incomplete"
        p = self.incomplete_root / rel
        return p.with_name(f"{p.name}.{etag}.incomplete")

    def incomplete_bytes(self, rel: str, etag: str | None) -> int:
        """Bytes already written to the temp file. Both layouts end with '{etag}.incomplete'."""
        if not etag:
            return 0
        if self.local_dir is None:
            try:
                return (self.blobs / f"{etag}.incomplete").stat().st_size
            except OSError:
                return 0
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
        try:
            st = self.final_path(rel).stat()  # follows the snapshot symlink
            if size == 0 or st.st_size == size:
                return st.st_size
        except OSError:
            pass
        return None

    def link_into_snapshot(self, rel: str, etag: str) -> None:
        """Recreate what huggingface_hub would: snapshots/<rev>/<rel> -> ../../blobs/<etag>."""
        if self.local_dir is not None:
            return
        target = self.snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            return
        rel_to_blob = os.path.relpath(self.blobs / etag, target.parent)
        os.symlink(rel_to_blob, target)

    def write_ref(self, revision: str) -> None:
        if self.local_dir is not None:
            return
        self.refs.mkdir(parents=True, exist_ok=True)
        (self.refs / "main").write_text(revision)


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


# --- ModelScope backend -------------------------------------------------------------------


def ms_download_file(repo_id: str, revision: str, f: dict, loc: Locator, stop: threading.Event) -> None:
    """Range-resumed single-stream download into the HF cache blob for this file's etag."""
    rel, size = f["path"], f["size"]
    etag = f["etag"] or f"ms-{hashlib.sha1(rel.encode()).hexdigest()}"  # provisional for non-LFS
    if loc.done_bytes(rel, f["etag"], size) is not None:
        if f["etag"]:
            loc.link_into_snapshot(rel, f["etag"])
        return

    tmp = loc.incomplete_path(rel, etag)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    url = ms_file_url(repo_id, rel, revision)

    with tmp.open("ab") as fh:
        resume = fh.tell()
        headers = {"Range": f"bytes={resume}-"} if resume else {}
        with httpx.Client(timeout=60.0, follow_redirects=True) as c:
            with c.stream("GET", url, headers=headers) as r:
                if resume and r.status_code == 200:
                    # Server ignored Range — restart cleanly rather than corrupt the file.
                    fh.close()
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"{rel}: server ignored Range, will retry from 0")
                r.raise_for_status()
                for chunk in r.iter_bytes(1 << 20):
                    if stop.is_set():
                        return
                    fh.write(chunk)

    if size and tmp.stat().st_size != size:
        raise RuntimeError(f"{rel}: got {tmp.stat().st_size} bytes, expected {size}")

    # Verify against the hash ModelScope published (same value HF uses for LFS blobs).
    if f.get("ms_sha256") and f.get("lfs"):
        h = hashlib.sha256()
        with tmp.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != f["ms_sha256"]:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{rel}: sha256 mismatch, discarded")

    if loc.local_dir is not None:
        dest = loc.final_path(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(dest))
        return

    # Non-LFS files are named by git blob id in the HF cache; compute it so the blob is reusable.
    final_etag = f["etag"] or git_blob_sha1(tmp)
    f["etag"] = final_etag
    blob = loc.blobs / final_etag
    blob.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(blob))
    loc.link_into_snapshot(rel, final_etag)


def ms_download(repo_id: str, revision: str, files, loc: Locator, workers: int, stop: threading.Event):
    errors: list[str] = []

    def one(f):
        try:
            ms_download_file(repo_id, revision, f, loc, stop)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}"[:200])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, files))
    if errors:
        raise RuntimeError("; ".join(errors[:3]))


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
    ap.add_argument("--source", default="auto", choices=["auto", "hf", "modelscope"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tries", type=int, default=200)
    ap.add_argument("--state", default=None)
    ap.add_argument(
        "--probe",
        action="store_true",
        help="report proxy signals + measured source speeds and exit (no download)",
    )
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

    if args.probe:
        probe_state = State(Path(os.devnull), args.repo, args.type, "")
        probe_state.publish = lambda: None  # type: ignore[method-assign]
        chosen = pick_source(args.repo, args.type, args.source, probe_state)
        print(f"[probe] would use: {chosen}", flush=True)
        return 0

    if not args.state:
        ap.error("--state is required unless --probe is given")
    state_path = Path(args.state).expanduser()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    dest = args.local_dir or (args.cache_dir or "")
    state = State(state_path, args.repo, args.type, dest)
    state.set(status="resolving")
    state.publish()
    for n in ENV_NOTES:
        print(f"[env] {n}", flush=True)

    source = pick_source(args.repo, args.type, args.source, state)
    state.set(source=source)

    dl_rev = args.revision or "master"
    try:
        if source == "modelscope":
            rev, dl_rev, all_files = ms_list_files(args.repo, dl_rev)
        else:
            rev, all_files = hf_list_files(HfApi(), args.repo, args.type, args.revision)
    except Exception as e:  # noqa: BLE001
        state.set(status="error", error=f"could not list repo via {source}: {e}")
        state.publish()
        print(f"[fatal] {e}", file=sys.stderr, flush=True)
        return 2

    files = apply_patterns(all_files, args.include, args.exclude)
    if not files:
        state.set(status="error", error="no files matched --include/--exclude")
        state.publish()
        return 2

    loc = Locator(args.repo, args.type, rev, args.local_dir, args.cache_dir)
    if not loc.local_dir:
        state.set(dest=str(loc.snapshot))
        loc.blobs.mkdir(parents=True, exist_ok=True)
        loc.write_ref(rev)

    # Record what we start from, so the window can show "resumed" honestly.
    resumed = 0
    for f in files:
        if loc.done_bytes(f["path"], f["etag"], f["size"]) is None:
            f["resumed_from"] = loc.incomplete_bytes(f["path"], f["etag"])
            resumed += f["resumed_from"]

    total = sum(f["size"] for f in files)
    state.set(status="running", total_bytes=total, revision=rev, resumed_bytes=resumed)
    state.publish()
    print(
        f"[start] {args.repo} via {source} · {len(files)} files · {total / 1e9:.2f} GB"
        + (f" · resuming from {resumed / 1e6:.1f} MB on disk" if resumed else ""),
        flush=True,
    )
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

    hf_kwargs = dict(
        repo_id=args.repo,
        repo_type=args.type,
        revision=args.revision,
        allow_patterns=args.include or None,
        ignore_patterns=args.exclude or None,
        max_workers=max(1, args.workers),
    )
    if args.local_dir:
        hf_kwargs["local_dir"] = args.local_dir
    if args.cache_dir:
        hf_kwargs["cache_dir"] = args.cache_dir

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
            if source == "modelscope":
                ms_download(args.repo, dl_rev, files, loc, args.workers, stop)
            else:
                snapshot_download(**hf_kwargs)
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
    missing = [f["path"] for f in files if loc.done_bytes(f["path"], f["etag"], f["size"]) is None]
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
        f"[done] {args.repo} via {source} → {state.data['dest']}"
        if not missing
        else f"[error] incomplete: {missing[:5]}",
        flush=True,
    )
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
