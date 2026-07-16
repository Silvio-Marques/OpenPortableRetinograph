"""
Removal of LED glare from retinal images captured by the OWL device.

Author: Dhruv Joshi

Removes the two direct specular reflections produced by the 20D lens in
OWL 1.0 (part of the openDR project, Srujana Center for Innovation,
LV Prasad Eye Institute, Hyderabad, India).
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Final

from .gpu_config import USE_GPU

# ---------------------------------------------------------------------------
# Glare-region constants (pixels in a full-resolution OWL frame)
# ---------------------------------------------------------------------------

# Centre of the glare patch.
_GLARE_X: Final[int] = 1396
_GLARE_Y: Final[int] = 958

# Half-width of the square crop used to detect and inpaint the glare.
_HALF_WIDTH: Final[int] = 100

# Fraction of the maximum pixel value above which a pixel is considered
# saturated (part of the specular highlight).
_SAT_THRESHOLD: Final[float] = 0.9

# Pre-computed dilation kernel.  MORPH_RECT allows OpenCV to use a
# separable row+column decomposition — O(n·2k) instead of O(n·k²).
_DILATE_KERNEL: Final[np.ndarray] = cv2.getStructuringElement(
    cv2.MORPH_RECT, (25, 25)
)


def remove_glare(im: np.ndarray, saturation_threshold: float = _SAT_THRESHOLD) -> np.ndarray:
    """Remove direct specular LED reflections from a retinal image.

    Generates a binary mask by thresholding the green channel within a
    square region centred on the known glare location, dilates the mask
    to cover the full reflection area, then inpaints the affected region
    using the Telea algorithm.

    The operation is performed on a cropped *view* of *im* to avoid
    copying the entire full-resolution frame, which matters on
    memory-constrained Raspberry Pi hardware.  The array is modified in
    place.  The dilation kernel is pre-computed as a module-level constant
    (``_DILATE_KERNEL``) using ``cv2.MORPH_RECT`` so OpenCV applies a
    separable decomposition on each call.

    When :data:`~modules.gpu_config.USE_GPU` is ``True`` (OpenCL
    available), the dilation mask is uploaded to the GPU as a
    :class:`cv2.UMat` before ``cv2.dilate`` and downloaded back to a
    NumPy array before ``cv2.inpaint``, which does not support UMat.  On a
    stock Raspberry Pi 4 (no functional OpenCL driver) ``USE_GPU`` is
    ``False`` and the path is identical to the original CPU-only
    implementation.

    Parameters
    ----------
    im:
        Full-resolution BGR retinal image with dtype ``uint8``.  Modified
        in place.

    Returns
    -------
    np.ndarray
        The same array passed in (modified in place) with the specular
        glare region inpainted.
    """
    y0, y1 = _GLARE_Y - _HALF_WIDTH, _GLARE_Y + _HALF_WIDTH
    x0, x1 = _GLARE_X - _HALF_WIDTH, _GLARE_X + _HALF_WIDTH

    # Threshold the green channel of the crop to locate saturated pixels.
    green_crop: np.ndarray = im[y0:y1, x0:x1, 1]
    effective_threshold = float(np.clip(saturation_threshold, 0.0, 1.0))
    _ret, temp_mask = cv2.threshold(
        green_crop, effective_threshold * 256, 255, cv2.THRESH_BINARY
    )

    temp_mask = cv2.dilate(
        cv2.UMat(temp_mask) if USE_GPU else temp_mask,
        _DILATE_KERNEL,
    )

    # cv2.inpaint does not support UMat; download from the OpenCL device
    # before calling it.
    cpu_mask: np.ndarray = temp_mask.get() if USE_GPU else temp_mask  # type: ignore[union-attr]

    # Inpaint only the small crop rather than the whole frame.
    im[y0:y1, x0:x1, :] = cv2.inpaint(
        im[y0:y1, x0:x1, :], cpu_mask, 1, cv2.INPAINT_TELEA
    )

    return im
