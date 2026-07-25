---
name: hf-resume
description: Get a Hugging Face model or dataset onto local disk — the default way to download ANY HF repo, replacing plain `hf download` / `snapshot_download`. Use whenever the user wants weights or data fetched: "下载 <repo-id> 这个模型", "把这个模型/数据集拉下来", "download <org/name> from HuggingFace", "get the weights for X onto my disk", "pull this repo", "grab this checkpoint" — and especially when a download broke partway and must continue: "下到一半断了，接着下", "续传不管用", "it downloaded several GB and started over", "resume the interrupted download". Multi-GB repos and interrupted downloads are the core case. Downloads via a pinned huggingface_hub==1.16.4 run by uv (newer versions cannot resume across runs) and opens a native macOS Swift progress window with an overall bar, speed, ETA, the full file list, and per-file live progress. Also picks the source automatically — direct HuggingFace when it is reachable and fast (e.g. a proxy/VPN is up), the ModelScope mirror when it is not (走直连 or 走国内镜像) — so it keeps working with or without a proxy. NOT for `hf auth`/upload/repo admin or the `hf` subcommand reference (that is hf-cli), NOT for downloading a plain URL/tarball/release asset/video (that is filetransfer), NOT for choosing which model to use (huggingface-best) or estimating its memory (hf-mem).
---

# hf-resume

`hf download` **cannot resume across runs.** Since ~`huggingface_hub` 1.17 each attempt writes
to `<blob>.<random8>.incomplete` and deletes it in a `finally` — the library's own comment says
the partial "could not be reused anyway since the temporary name is unique to this download".
Kill a 17 GB download at 90% and you start again at byte 0.

This skill downloads through a **pinned `huggingface_hub==1.16.4`** (run by `uv`), the last line
that opens a deterministic `<etag>.incomplete` and passes `resume_size` to `http_get` so a real
`Range` request continues the file. A **native Swift window** shows progress.

## Use it

```bash
S=~/.claude/skills/hf-resume/scripts/hf-resume     # or wherever this skill is installed

$S pull mlx-community/flux2-klein-9b-8bit          # detached + window; source auto-picked
$S pull Qwen/Qwen3-8B --include '*.safetensors' --workers 4
$S pull HuggingFaceFW/fineweb --type dataset --local-dir ./fineweb
$S pull <repo> --modelscope   # force the CN mirror  (--hf forces direct)
$S status            # text progress (any time, any shell)
$S window            # re-open the window
$S stop              # stop; partials stay resumable
$S list              # every run
$S verify <repo>     # checksum-verify what landed
$S doctor [repo]     # environment check + which source it would pick, and why
```

Re-running the **same `pull`** is how you resume. It is idempotent: finished files are skipped,
half-finished ones continue from their byte offset, and the window reports
`resumed from N MB already on disk`.

Downloads land in the **HF cache** by default, so `mflux` / `diffusers` / `mlx` resolve them by
repo id with no path juggling. `--local-dir` gives a flat copy instead.

The daemon is **detached**: it keeps running after the command returns, after the window is
closed, and after this session ends. `status` and `window` reattach to it.

## Sources: direct or ModelScope, and why the choice is measured

`--source auto` (default) probes direct HuggingFace and uses it whenever it is reachable at
≥2 MB/s. Only when HF is unreachable or crawling does it probe **ModelScope** and switch.
Override with `--hf` / `--modelscope`, or `hf-resume doctor <repo>` to see the decision.

**Do not try to detect the proxy instead.** Clash/mihomo in TUN mode routes at the network
layer: `scutil --proxy` reports `HTTPEnable: 0` and no proxy env vars exist, while
huggingface.co is in fact fast. Config-based detection reports "no proxy" exactly when direct
is the right answer. Proxy signals are logged as *explanation*; the measurement decides.

Short probes are noisy in both directions — TCP slow start undershoots (a 3 MB read measured
1.2 MB/s on a link sustaining 8 MB/s), a warm CDN edge overshoots. So the probe skips its
first 1.5 MB and its verdict is only ever "usable vs not", never a close race.

**Either source fills the same cache.** For LFS files the HF cache names each blob by its
sha256, and ModelScope publishes the same sha256 for the same repo (verified byte-identical,
and non-LFS files get their git-blob-sha1 computed locally). So ModelScope bytes are valid HF
blobs: after a ModelScope pull, an `--hf` pull of the same repo finishes in seconds and
re-downloads nothing — measured, with the blob's mtime unchanged. Switch freely.

## Rules — each one is a bug someone already shipped

1. **Never `aria2c -x8` / any multi-connection downloader on Hugging Face.** A signed CDN URL
   pins one byte range in its CloudFront policy; reusing that URL for a different range returns
   **403**. With preallocation the file ends up full-size **with holes** — silent corruption that
   only a checksum catches. Keep every file single-stream. Concurrency comes from `--workers`
   (several files at once), never from splitting one file.
2. **Xet must stay disabled** (`HF_HUB_DISABLE_XET=1`, set by the downloader). Even in 1.16.4,
   `resume_size` is passed to `http_get` but **never** to `xet_get`, while the incomplete file is
   opened `"ab"` — so on a Xet repo a leftover partial can get a full reconstruction appended
   onto it. Xet is faster when it works, but it cannot resume here.
3. **`hf_transfer` cannot resume at all.** `HF_HUB_ENABLE_HF_TRANSFER` is stripped.
4. **Mirrors that redirect are worse than nothing.** `hf-mirror.com` now 308-redirects to
   `huggingface.co` (zero acceleration) and breaks modern `hf` metadata resolution outright
   (`Local entry not found. Distant resource does not seem to be on huggingface.co`). So an
   `HF_ENDPOINT` that does not point at `huggingface.co` is ignored **and reported**, never
   silently dropped. A genuinely working mirror is still worth having: pass `--endpoint URL`
   (wins unconditionally) or set `HF_RESUME_KEEP_ENDPOINT=1` to keep the ambient one. If
   `doctor` finds a stale mirror in `~/.zshrc`, offer to delete that line.
5. **Progress comes from the filesystem, not tqdm.** tqdm's `desc` is the CDN basename truncated
   to 40 chars, and one repo can hold several files named `0.safetensors`. The downloader maps
   each file to its `{etag}.incomplete` instead — exact for both cache and `--local-dir`.
6. **`--force-download` throws away resumable bytes.** Only reach for it when a checksum fails.

## Shape

```
scripts/hf-resume      bash dispatch; detaches the daemon, compiles+launches the window
scripts/downloader.py  uv PEP-723 script pinning huggingface_hub==1.16.4; publishes state.json
scripts/Window.swift   SwiftUI reader of state.json (compiled on first use, cached by hash)
```

Runtime state lives in `~/.cache/hf-resume/runs/<slug>/` (`state.json`, `downloader.log`, `pid`).
`state.json` is written atomically (temp + rename), so the window never reads a torn file.

The window is a **pure reader** and owns no download — closing it stops nothing.

Two things about it are load-bearing and were found the hard way:
- It sets `NSApp.setActivationPolicy(.regular)`; a non-bundled binary is `.prohibited` by default
  and would never show a window.
- It creates its `NSWindow` **explicitly** and hosts the SwiftUI tree in an `NSHostingView`,
  rather than using a SwiftUI `App`/`Window` scene. In a bare binary a Scene works while the App
  holds only value `@State`, but the moment the App owns a model object (`@StateObject` on an
  ObservableObject, or `@State` on an `@Observable` class, with or without `@MainActor`) the
  process launches, idles in a healthy event loop, and creates **zero** windows — no crash, no
  log. Because it is top-level code it must be compiled **without** `-parse-as-library`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Starts from 0 despite this skill | Partials from modern `hf` carry a random suffix and are unusable. `hf cache prune`, then pull again. `doctor` counts them and their wasted GB. |
| `Local entry not found. Distant resource does not seem to be on huggingface.co` | A redirecting `HF_ENDPOINT`. Run `doctor`; remove the line it points at in `~/.zshrc`. |
| Window never appears | `swiftc` missing → `xcode-select --install`. Check the compile error on stderr; `status` still works headless. |
| Speed is the problem, not resume | Nothing here fixes bandwidth. `status` shows the honest rate; the daemon grinds on and speeds up by itself if the link improves. |
| Checksum failure | A previous multi-connection download probably holed the file. Delete that file and pull again; never trust a same-size file after aria2 touched it. |
| Repo is gated | Log in with `hf auth login` (that is `hf-cli` territory); the pinned lib reads the same token. |
