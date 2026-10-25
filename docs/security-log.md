# Security Log

English | [繁體中文](security-log.zh-TW.md)

- Date: 2026-10-25
- Branch: `fix/security-upgrades`
- Scope: `pyproject.toml`, `uv.lock`, `groundingdino/`

## Summary

PyTorch moved to the CUDA 12.6 index, gradio and transformers were upgraded, and the bundled GroundingDINO code was adjusted for transformers 5 and current supervision. pillow still has open advisories because gradio caps its version.

## Changes

| Item | Before | After | Reason |
|---|---|---|---|
| torch / torchvision | 2.5.1+cu121 / 0.20.1 | 2.14.1+cu126 / 0.29.1 | CVE-2025-32434 (`torch.load(weights_only=True)` remote code execution) and later memory-corruption advisories |
| gradio | 5.23.1 | 6.30.0 | File read, SSRF and path traversal advisories (fixed in 6.16) |
| transformers | 4.50.1 | 5.20.2 | CVE-2026-5241 and earlier advisories |
| `groundingdino/models/GroundingDINO/bertwarper.py` | - | local fallbacks | transformers 5 removed the BERT mask helpers |
| `groundingdino/util/inference.py` `annotate()` | `BoxAnnotator(labels=...)` | `BoxAnnotator` + `LabelAnnotator` | supervision no longer accepts `labels` in `BoxAnnotator`; this was already broken with the previously locked 0.25.1 |

## What you need to know

- PyTorch now comes from the CUDA 12.6 wheel index (`pytorch-cu126`) instead of cu121, because the cu121 index stops at torch 2.5.1. You need an NVIDIA driver that supports CUDA 12.6 or newer. Recreate the virtual environment so the new wheels are installed.

## Remaining issues

### Dependency versions held back upstream

| Package | Version | Advisories | Why it is not fixed | Impact and mitigation |
|---|---|---|---|---|
| pillow | 12.3.0 | 41 (fixed in 13.0.0) | gradio >= 6.16 requires `pillow<13`; no gradio release allows Pillow 13 yet. | Crafted image files (for example TIFF or PDF) can crash the process or read out-of-bounds memory when decoded. Run the Gradio app locally only (no `share=True`, no public port) and open only images you trust. Upgrade once gradio allows Pillow 13. |

## How it was verified

- osv-scanner on `uv.lock`; only pillow remains.
- CPU, torch 2.14.1 and transformers 5.20.2: GroundingDINO builds from its config and `predict()` / `annotate()` run.
- The Gradio UI builds and serves with the model modules replaced by stubs.

## Not verified

- Inference on a GPU with the real model weights (the test environment has no GPU and no weights).
