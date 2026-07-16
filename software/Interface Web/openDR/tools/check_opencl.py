#!/usr/bin/env python3
"""
OpenCL availability diagnostic for the openDR pipeline.

Prints a summary of OpenCL support on the current machine and reports
whether the openDR GPU offload path (``modules.gpu_config.USE_GPU``) is
active.

Usage::

    python tools/check_opencl.py

Expected output on a stock Raspberry Pi 4 (Bookworm / Bullseye)::

    OpenCV version : 4.x.x
    haveOpenCL()   : False
    USE_GPU        : False
    OPENDR_USE_GPU : (not set)

    VideoCore VI (Raspberry Pi 4) notes
    ------------------------------------
    * VC4CL targets VideoCore IV (Pi 1-3) only.
    * Mesa Clover (V3D) is experimental; haveOpenCL() typically returns
      False on a stock OS install as of 2025-07.
    * Set OPENDR_USE_GPU=1 to force-enable the T-API code path for testing
      (operations will silently fall back to CPU if no driver is present).
"""
from __future__ import annotations

import os
import sys

import cv2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yn(value: bool) -> str:
    return "True  ✓" if value else "False ✗"


def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# ---------------------------------------------------------------------------
# Basic availability
# ---------------------------------------------------------------------------

print("=" * 60)
print("openDR — OpenCL / GPU offload diagnostic")
print("=" * 60)

print(f"\nOpenCV version : {cv2.__version__}")
have_ocl = cv2.ocl.haveOpenCL()
print(f"haveOpenCL()   : {_yn(have_ocl)}")

env_val = os.environ.get("OPENDR_USE_GPU", "")
env_display = repr(env_val) if env_val else "(not set)"
print(f"OPENDR_USE_GPU : {env_display}")

# Import after printing env so the module picks up any override.
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.gpu_config import USE_GPU  # noqa: E402
    print(f"USE_GPU        : {_yn(USE_GPU)}")
except Exception as exc:  # pragma: no cover
    print(f"USE_GPU        : (import error: {exc})")
    USE_GPU = False

# ---------------------------------------------------------------------------
# OpenCL platform / device enumeration
# ---------------------------------------------------------------------------

if have_ocl:
    _print_section("OpenCL platforms and devices")
    try:
        # cv2.ocl is a C++ binding; enumerate via a minimal UMat round-trip
        # and rely on cv2.getBuildInformation for device details.
        cv2.ocl.setUseOpenCL(True)
        info_lines = [
            ln.strip()
            for ln in cv2.getBuildInformation().splitlines()
            if "OpenCL" in ln or "opencl" in ln.lower()
        ]
        for ln in info_lines:
            print(" ", ln)
    except Exception as exc:
        print(f"  (enumeration failed: {exc})")
else:
    _print_section("OpenCL platforms and devices")
    print("  (no OpenCL runtime detected)")

# ---------------------------------------------------------------------------
# Functional round-trip test
# ---------------------------------------------------------------------------

_print_section("Functional T-API round-trip test")

import numpy as np  # noqa: E402

test_img = np.random.randint(0, 256, (804, 804), dtype=np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

try:
    cv2.ocl.setUseOpenCL(True)
    u = cv2.UMat(test_img)
    result_u = cv2.erode(u, kernel, iterations=1)
    result_cpu = result_u.get()

    cpu_reference = cv2.erode(test_img, kernel, iterations=1)
    match = np.array_equal(result_cpu, cpu_reference)
    print(f"  UMat erode (804×804): {'PASS ✓' if match else 'FAIL — result differs from CPU reference'}")
except Exception as exc:
    print(f"  UMat erode (804×804): FAIL — {exc}")

# ---------------------------------------------------------------------------
# Raspberry Pi notes
# ---------------------------------------------------------------------------

_print_section("Raspberry Pi 4 / VideoCore VI notes")
print(
    "  * VC4CL targets VideoCore IV (Pi 1-3) — not compatible with Pi 4/5.\n"
    "  * Mesa Clover (V3D OpenCL) is experimental; haveOpenCL() typically\n"
    "    returns False on a stock Raspberry Pi OS install (as of 2025-07).\n"
    "  * When USE_GPU is False, the openDR pipeline runs CPU-only — identical\n"
    "    behaviour to before the T-API integration.\n"
    "  * To test the UMat code path on a non-Pi machine with OpenCL, run:\n"
    "      OPENDR_USE_GPU=1 python tools/check_opencl.py\n"
    "  * cv2.inpaint and cv2.findContours do not support UMat and always run\n"
    "    on CPU regardless of USE_GPU."
)

print()
