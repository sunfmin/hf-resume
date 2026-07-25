# hf-resume

A Claude Code skill for **Hugging Face downloads that actually resume**, with a native macOS
progress window.

## The problem

`hf download` cannot resume across runs. Since ~`huggingface_hub` 1.17, every attempt writes to
`<blob>.<random8>.incomplete` and deletes it in a `finally`. The library's own comment:

> On failure, do not keep a partial file around: **it could not be reused anyway since the
> temporary name is unique to this download.**

`http_get` is also called without `resume_size`, so each run starts at byte 0. Interrupt a 17 GB
model at 90% and you download all 17 GB again.

## What this does

- Downloads through a pinned **`huggingface_hub==1.16.4`** (run by `uv`) — the last line that
  opens a deterministic `<etag>.incomplete` and passes `resume_size` to `http_get`, producing a
  real `Range` request that continues the file.
- Opens a **native Swift window**: overall progress bar, speed, ETA, the full file list, and
  per-file live progress. It is a pure reader of a JSON state file, so closing it stops nothing.
- Runs **detached** (`os.setsid()`): survives the command returning, the window closing, the
  terminal going away, and its launcher being killed.
- **Retries forever by default** — resume makes a retry nearly free, so a flaky link eventually
  finishes instead of starting over.

```bash
hf-resume pull mlx-community/flux2-klein-9b-8bit    # detached + window
hf-resume status                                    # progress in text
hf-resume stop                                      # partials stay resumable
hf-resume verify <repo>                             # checksum what landed
hf-resume doctor                                    # environment sanity check
```

Re-running the same `pull` is how you resume.

## Three ways to corrupt a Hugging Face download (all avoided here)

1. **Multi-connection downloaders.** A signed HF CDN URL pins one byte range in its CloudFront
   policy — reusing it for a different range returns **403**. `aria2c -x8` preallocates, so the
   file ends up full-size **with holes**: silent corruption only a checksum catches. Every file
   here stays single-stream; concurrency comes from downloading several files at once.
2. **Xet + a leftover partial.** Even in 1.16.4, `resume_size` reaches `http_get` but never
   `xet_get`, while the incomplete file is opened `"ab"` — a partial can get a full
   reconstruction appended onto it. `HF_HUB_DISABLE_XET=1` is forced.
3. **`hf_transfer`.** No resume at all. Stripped from the environment.

Bonus: `hf-mirror.com` now 308-redirects to `huggingface.co` (zero acceleration) *and* breaks
modern `hf` metadata resolution. Any non-`huggingface.co` `HF_ENDPOINT` is ignored; `doctor`
tells you where it came from.

## Install

```bash
npx skills add sunfmin/hf-resume -g
```

Requires `uv`, and Xcode command line tools for `swiftc` (the window compiles once, ~0.7 s, and
is cached by source hash).

## Layout

```
skills/hf-resume/SKILL.md              the skill
skills/hf-resume/scripts/hf-resume     CLI: pull / status / window / stop / list / verify / doctor
skills/hf-resume/scripts/downloader.py uv PEP-723 script, pinned lib, publishes state.json
skills/hf-resume/scripts/Window.swift  SwiftUI reader of state.json
```
