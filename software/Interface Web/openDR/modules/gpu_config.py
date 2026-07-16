"""
GPU / OpenCL configuration for the openDR image-processing pipeline.

Sets the ``USE_GPU`` flag and enables the OpenCV T-API (``cv2.UMat`` /
OpenCL back-end) when a compatible OpenCL device is detected at import
time.  Callers that want to force CPU-only mode can set the environment
variable ``OPENDR_USE_GPU=0`` before importing any openDR module.

Raspberry Pi 4 / VideoCore VI notes
------------------------------------
The upstream VC4CL driver targets VideoCore **IV** (Raspberry Pi 1–3)
only.  Mesa Clover (V3D OpenCL) is still experimental on Raspberry Pi OS
Bookworm / Bullseye — ``cv2.ocl.haveOpenCL()`` typically returns ``False``
on a stock installation.

When ``USE_GPU`` is ``False`` (the expected case on Pi 4 with the stock OS)
all ``cv2.UMat`` conversions are skipped and the pipeline runs entirely on
CPU, identical to the behaviour before this change.

If a future driver update makes ``haveOpenCL()`` return ``True``, or if
the system has an Intel/AMD GPU (e.g. during development on an x86
machine), the T-API is enabled automatically with no code changes.

Environment variable
--------------------
``OPENDR_USE_GPU``
    * ``"1"`` / ``"true"`` / ``"yes"``  — force-enable, even when
      ``haveOpenCL()`` returns ``False`` (useful for testing, but
      operations will silently fall back to CPU).
    * ``"0"`` / ``"false"`` / ``"no"``  — force-disable, even when an
      OpenCL device is present.
    * *(unset / any other value)*        — auto-detect via
      ``cv2.ocl.haveOpenCL()``.
"""
from __future__ import annotations

import os

import cv2

# ---------------------------------------------------------------------------
# OpenCL auto-detection
# ---------------------------------------------------------------------------

# ``cv2.ocl.haveOpenCL()`` returns True only when a functional OpenCL
# runtime and at least one device are present.  On a stock Raspberry Pi 4
# running Bookworm/Bullseye this is expected to return False (VideoCore VI
# has no production OpenCL driver as of 2025-07).
_ocl_detected: bool = cv2.ocl.haveOpenCL()

# ---------------------------------------------------------------------------
# Environment-variable override
# ---------------------------------------------------------------------------

_env_raw: str = os.environ.get("OPENDR_USE_GPU", "").strip().lower()
if _env_raw in ("1", "true", "yes"):
    USE_GPU: bool = True
elif _env_raw in ("0", "false", "no"):
    USE_GPU = False
else:
    USE_GPU = _ocl_detected

# ---------------------------------------------------------------------------
# Propagate to OpenCV's global OpenCL switch
# ---------------------------------------------------------------------------

# Setting ``cv2.ocl.setUseOpenCL(True)`` when no driver is available is a
# no-op — OpenCV will automatically fall back to CPU for every call.
cv2.ocl.setUseOpenCL(USE_GPU)
