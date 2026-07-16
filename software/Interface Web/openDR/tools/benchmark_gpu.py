#!/usr/bin/env python3
"""
CPU vs OpenCL T-API benchmark for the openDR image-processing pipeline.

Measures the per-step and total wall-clock time of the fundus extraction
and glare-removal pipeline under two modes:

* **CPU** — OpenCL T-API disabled (``OPENDR_USE_GPU=0``).
* **T-API** — OpenCL T-API enabled if ``cv2.ocl.haveOpenCL()`` returns
  ``True``, or forced via ``OPENDR_USE_GPU=1``.  When no OpenCL driver is
  present the T-API path silently falls back to CPU; both numbers will be
  identical in that case.

A synthetic OWL-sized BGR image (2772 × 1848 px) is generated as the test
fixture so no real camera image is required.

Usage::

    # Auto-detect (recommended — shows GPU column only when OpenCL is present):
    python tools/benchmark_gpu.py

    # Force both paths even without a real OpenCL device (for code-path testing):
    python tools/benchmark_gpu.py --force-gpu

    # More iterations for stable statistics:
    python tools/benchmark_gpu.py --iterations 20

    # Use a real captured image instead of the synthetic fixture:
    python tools/benchmark_gpu.py --image /path/to/owl_capture.jpg

Pipeline steps measured
-----------------------
1. ``extract_circles``  — circular ROI masking
2. ``erode_thresh``     — cvtColor + crop + downscale + threshold + erode×5 +
                          GaussianBlur + upscale  *(GPU-accelerated portion)*
3. ``ellipse_fit``      — contour detection + fitEllipse + mask
4. ``remove_glare``     — green-channel threshold + dilate + inpaint
                          *(GPU-accelerated dilate)*
5. **Total pipeline**   — wall-clock for the complete ``grade`` pre-processing
                          (steps 1–4)

Results
-------
On a stock Raspberry Pi 4 (no OpenCL driver) both columns will be
identical because the T-API silently falls back to CPU.  On a machine
with a working OpenCL device (e.g. an Intel integrated GPU on an x86
development box) the T-API column should show speed-up in ``erode_thresh``
and ``remove_glare``.

Expected gains when OpenCL is functional
-----------------------------------------
* ``erode_thresh``  : 2–4× (``cv2.erode`` ×5 is highly parallel at 804×804)
* ``remove_glare``  : 1.5–2.5× for ``cv2.dilate`` on the 200×200 crop
* ``ellipse_fit``   : no change (``cv2.findContours`` / ``fitEllipse`` CPU-only)
* ``extract_circles``: no change (``cv2.bitwise_and`` already fast on CPU)
* **Total**         : 10–30% reduction in pre-processing wall-clock
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Callable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Benchmark openDR pipeline: CPU vs OpenCL T-API",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--iterations",
    type=int,
    default=10,
    metavar="N",
    help="Number of timed iterations per mode (default: 10)",
)
parser.add_argument(
    "--warmup",
    type=int,
    default=2,
    metavar="W",
    help="Warm-up iterations before timing (default: 2)",
)
parser.add_argument(
    "--image",
    metavar="PATH",
    help="Path to a real OWL fundus JPEG (default: use synthetic fixture)",
)
parser.add_argument(
    "--force-gpu",
    action="store_true",
    help=(
        "Force-enable the T-API code path even when haveOpenCL() is False "
        "(useful to test the UMat code path; ops fall back to CPU silently)"
    ),
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path regardless of where the script runs from
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Synthetic OWL-sized test image
# ---------------------------------------------------------------------------

# OWL camera constants (must match modules/extract.py):
#   centre  = (1386, 948),  radius = 804
# The image must be at least 2190 px wide and 1752 px tall.
_OWL_W, _OWL_H = 2772, 1848
_OWL_CENTER = (1386, 948)
_OWL_RADIUS = 804


def _build_synthetic_image() -> np.ndarray:
    """Return a plausible synthetic OWL fundus BGR image."""
    img = np.zeros((_OWL_H, _OWL_W, 3), dtype=np.uint8)
    # Fill the circular region with a mid-brightness gradient.
    for c in range(3):
        channel = img[:, :, c]
        cy, cx = np.mgrid[0:_OWL_H, 0:_OWL_W]
        dist = np.sqrt((cx - _OWL_CENTER[0]) ** 2 + (cy - _OWL_CENTER[1]) ** 2)
        channel[dist <= _OWL_RADIUS] = np.clip(
            100 - (dist[dist <= _OWL_RADIUS] / _OWL_RADIUS * 40).astype(np.uint8),
            0, 255,
        ).astype(np.uint8)
    # Simulate a bright glare spot near the known glare centre.
    cv2.circle(img, (1396, 958), 40, (240, 255, 240), thickness=-1)
    return img


def _load_image() -> np.ndarray:
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"ERROR: cannot read image file: {args.image!r}")
        return img
    return _build_synthetic_image()


# ---------------------------------------------------------------------------
# Step-level timing helpers
# ---------------------------------------------------------------------------

from modules.extract import (  # noqa: E402
    ellipse_fit,
    erode_thresh,
    extract_circles,
)
from modules import remove_glare as rg_module  # noqa: E402


def _time_steps(image: np.ndarray) -> dict[str, float]:
    """Run all pipeline steps and return per-step wall-clock seconds."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    circle = extract_circles(image)
    timings["extract_circles"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    threshed = erode_thresh(circle)
    timings["erode_thresh"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    extracted = ellipse_fit(circle, threshed)
    timings["ellipse_fit"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rg_module.remove_glare(extracted)
    timings["remove_glare"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    return timings


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _run_mode(
    label: str,
    use_gpu: bool,
    image: np.ndarray,
    n_warmup: int,
    n_iters: int,
) -> dict[str, list[float]]:
    """Run the pipeline *n_iters* times and collect per-step timings."""
    import modules.gpu_config as gcfg

    gcfg.USE_GPU = use_gpu
    cv2.ocl.setUseOpenCL(use_gpu)

    # Also patch module-level USE_GPU in extract and remove_glare so the
    # conditional branches see the updated value.
    import modules.extract as ext_mod
    import modules.remove_glare as rg_mod

    ext_mod.USE_GPU = use_gpu  # type: ignore[attr-defined]
    rg_mod.USE_GPU = use_gpu  # type: ignore[attr-defined]

    print(f"\n[{label}] warm-up ({n_warmup} iter)...", end=" ", flush=True)
    for _ in range(n_warmup):
        _time_steps(image.copy())
    print("done")

    print(f"[{label}] timing ({n_iters} iter)...", end=" ", flush=True)
    results: dict[str, list[float]] = {
        k: [] for k in ("extract_circles", "erode_thresh", "ellipse_fit",
                         "remove_glare", "total")
    }
    for _ in range(n_iters):
        row = _time_steps(image.copy())
        for k, v in row.items():
            results[k].append(v)
    print("done")
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_MS = 1000.0  # seconds → milliseconds


def _fmt(values: list[float]) -> str:
    mean = statistics.mean(values) * _MS
    stdev = statistics.stdev(values) * _MS if len(values) > 1 else 0.0
    lo = min(values) * _MS
    hi = max(values) * _MS
    return f"{mean:7.1f} ± {stdev:5.1f}  [{lo:.1f}–{hi:.1f}]  ms"


def _print_table(
    cpu_data: dict[str, list[float]],
    gpu_data: dict[str, list[float]] | None,
) -> None:
    steps = ("extract_circles", "erode_thresh", "ellipse_fit",
             "remove_glare", "total")
    gpu_label = "T-API (GPU)" if args.force_gpu else (
        "T-API (GPU)" if cv2.ocl.haveOpenCL() else "T-API (no OCL)"
    )

    col_w = 44
    step_w = 18
    header_cpu = "CPU (ms: mean±stdev [min–max])"
    header_gpu = f"{gpu_label} (ms: mean±stdev [min–max])"
    sep = "-" * (step_w + 2 + col_w + (2 + col_w + 8 if gpu_data else 0))

    print()
    print("=" * len(sep))
    print("openDR Pipeline Benchmark Results")
    print(f"  image     : {args.image or 'synthetic OWL fixture (2772×1848)'}")
    print(f"  iterations: {args.iterations}  (+ {args.warmup} warm-up)")
    print(f"  OpenCL    : {cv2.ocl.haveOpenCL()}")
    print("=" * len(sep))

    fmt_hdr = f"{'Step':{step_w}}  {header_cpu:{col_w}}"
    if gpu_data:
        fmt_hdr += f"  {header_gpu:{col_w}}  Speed-up"
    print(fmt_hdr)
    print(sep)

    for step in steps:
        cpu_str = _fmt(cpu_data[step])
        row = f"{'  ' + step if step != 'total' else '  TOTAL':{step_w}}  {cpu_str:{col_w}}"
        if gpu_data:
            gpu_str = _fmt(gpu_data[step])
            cpu_mean = statistics.mean(cpu_data[step])
            gpu_mean = statistics.mean(gpu_data[step])
            speedup = cpu_mean / gpu_mean if gpu_mean > 0 else float("inf")
            marker = " ◀" if speedup >= 1.1 else ""
            row += f"  {gpu_str:{col_w}}  {speedup:5.2f}×{marker}"
        print(row)

    print(sep)
    if gpu_data:
        print()
        print("  ◀ = ≥10 % speed-up")
        if not cv2.ocl.haveOpenCL() and not args.force_gpu:
            print()
            print("  NOTE: No OpenCL device detected.  Both columns show CPU timings.")
            print("  To test with a real OpenCL device, run on an x86 machine with an")
            print("  Intel/AMD GPU, or wait for the Mesa V3D OpenCL driver to mature")
            print("  on Raspberry Pi OS.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading image...", end=" ", flush=True)
    image = _load_image()
    print(f"done  ({image.shape[1]}×{image.shape[0]} px BGR)")

    run_gpu_path = args.force_gpu or cv2.ocl.haveOpenCL()

    cpu_results = _run_mode(
        "CPU", use_gpu=False, image=image,
        n_warmup=args.warmup, n_iters=args.iterations,
    )

    gpu_results: dict[str, list[float]] | None = None
    if run_gpu_path:
        gpu_results = _run_mode(
            "T-API", use_gpu=True, image=image,
            n_warmup=args.warmup, n_iters=args.iterations,
        )

    _print_table(cpu_results, gpu_results)
