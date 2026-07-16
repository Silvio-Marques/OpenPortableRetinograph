"""
Image extraction utilities for the openDR fundus camera pipeline.

These functions isolate the retinal disc from a raw fundus photograph
captured by the OWL device and prepare it for downstream grading.

Performance notes (Raspberry Pi 4):
  * The circular ROI mask is built once per unique frame resolution and
    cached via :func:`_get_circle_mask` (``functools.lru_cache``).  The
    ``np.zeros`` + ``cv2.circle`` allocation no longer runs on every call.
  * ``_EROSION_KERNEL`` and ``_EROSION_KERNEL_HALF`` use
    ``cv2.getStructuringElement(cv2.MORPH_RECT, …)`` instead of
    ``np.ones``.  OpenCV recognises rectangular structuring elements as
    row/column-separable and decomposes the dilation/erosion into two 1-D
    passes, reducing per-pixel work from O(k²) to O(2k).
  * :func:`erode_thresh` crops to the axis-aligned bounding-box of the
    circular ROI before processing (~2.6 MP instead of ~5 MP, ≈2×
    faster), then works at half resolution (804×804 px) where the heavy
    morphological operations cost a further ≈4× less.  The result is
    upscaled back to ROI size and embedded in a full-size output so the
    public interface with :func:`ellipse_fit` is unchanged.
  * :func:`cv2.bitwise_and` replaces three separate per-channel
    ``np.multiply`` calls, reducing peak memory usage by ~3× per masking
    operation and delegating the work to an optimised C routine.
"""
from __future__ import annotations

import functools

import cv2
import numpy as np
from typing import Final

from .gpu_config import USE_GPU

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Pixel coordinates of the optical-axis centre in a full-resolution OWL frame.
_CENTER: Final[tuple[int, int]] = (1386, 948)

# Radius (px) of the circular region of interest.
_RADIUS: Final[int] = 804

# Brightness threshold used when binarising the grayscale image (empirical).
_THRESHOLD_VALUE: Final[int] = 65

# Square kernel side-length for morphological erosion (at full resolution).
_KERNEL_SIZE: Final[int] = 14

# Number of erosion passes applied to the thresholded image.
_EROSION_ITERATIONS: Final[int] = 5

# ---------------------------------------------------------------------------
# Derived ROI constants — axis-aligned bounding-box of the circular mask
# ---------------------------------------------------------------------------

_ROI_Y0: Final[int] = _CENTER[1] - _RADIUS   # 144
_ROI_Y1: Final[int] = _CENTER[1] + _RADIUS   # 1752
_ROI_X0: Final[int] = _CENTER[0] - _RADIUS   # 582
_ROI_X1: Final[int] = _CENTER[0] + _RADIUS   # 2190

# The bounding-box is a square of side 2 * _RADIUS = 1608 px.
_ROI_W: Final[int] = 2 * _RADIUS   # 1608
_ROI_H: Final[int] = 2 * _RADIUS   # 1608

# Half-resolution ROI dimensions used for the downscaled processing stage.
# 2 * _RADIUS / 2 == _RADIUS, so the half-size crop is exactly 804 × 804 px.
_ROI_HALF_W: Final[int] = _RADIUS   # 804
_ROI_HALF_H: Final[int] = _RADIUS   # 804

# ---------------------------------------------------------------------------
# Pre-computed kernels
# ---------------------------------------------------------------------------

# Full-resolution erosion kernel.  MORPH_RECT signals OpenCV to use the
# faster separable (row + column) decomposition — O(n·2k) vs O(n·k²).
_EROSION_KERNEL: Final[np.ndarray] = cv2.getStructuringElement(
    cv2.MORPH_RECT, (_KERNEL_SIZE, _KERNEL_SIZE)
)

# Half-resolution erosion kernel: 7×7 covers the same spatial area as 14×14
# at full scale, keeping morphological semantics equivalent after the 2×
# downscale applied inside :func:`erode_thresh`.
_EROSION_KERNEL_HALF: Final[np.ndarray] = cv2.getStructuringElement(
    cv2.MORPH_RECT, (_KERNEL_SIZE // 2, _KERNEL_SIZE // 2)
)

# ---------------------------------------------------------------------------
# Cached mask builder
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _get_circle_mask(h: int, w: int) -> np.ndarray:
    """Return a read-only circular ROI mask for a frame of *h* × *w* pixels.

    The result is cached so the ``np.zeros`` allocation and ``cv2.circle``
    draw happen only once per unique frame resolution across the lifetime of
    the process.  The returned array is marked read-only to prevent accidental
    mutation of the shared cache entry.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, _CENTER, _RADIUS, 255, thickness=-1)
    mask.flags.writeable = False
    return mask


def extract_circles(image: np.ndarray) -> np.ndarray:
    """Mask all pixels outside the known circular region of interest.

    A filled circular mask is drawn at the fixed centre/radius for the OWL
    optics.  Every pixel outside that circle is set to black ``(0, 0, 0)``
    while pixels inside keep their original BGR values.

    Performance (Raspberry Pi): the mask is a single-channel ``uint8`` array
    applied with :func:`cv2.bitwise_and`.  The mask itself is cached via
    :func:`_get_circle_mask` so the allocation runs at most once per unique
    frame resolution per process lifetime.

    Parameters
    ----------
    image:
        Source BGR image as a NumPy array with shape ``(H, W, 3)`` and
        dtype ``uint8``.

    Returns
    -------
    np.ndarray
        A new BGR image of the same shape and dtype as *image* with the
        region outside the circle zeroed out.
    """
    h, w = image.shape[:2]
    return cv2.bitwise_and(image, image, mask=_get_circle_mask(h, w))


def erode_thresh(image: np.ndarray, threshold_value: int = _THRESHOLD_VALUE) -> np.ndarray:
    """Convert *image* to a smoothed binary mask via threshold and erosion.

    The image is converted to grayscale, cropped to the axis-aligned
    bounding-box of the circular ROI, downscaled 2× for cheaper
    morphological operations, binarised using a fixed intensity threshold,
    eroded to remove small bright artefacts, then Gaussian-blurred to
    produce smoother contour boundaries for downstream ellipse fitting.
    The processed crop is upscaled back to ROI size and embedded in a
    full-size single-channel output so the caller interface is unchanged.

    Performance (Raspberry Pi):

    * Cropping to the ~2.6 MP ROI bounding-box before processing (instead
      of the full ~5 MP frame) halves the number of pixels passed to each
      operation.
    * A further 2× downscale to 804×804 px reduces the cost of
      ``cv2.erode`` × 5 and ``cv2.GaussianBlur`` by an additional ≈4×.
      The 7×7 half-scale kernel covers the same spatial neighbourhood as
      the 14×14 full-scale kernel.
    * ``_EROSION_KERNEL_HALF`` uses ``cv2.MORPH_RECT`` so OpenCV applies a
      separable row+column decomposition internally.
    * When :data:`~modules.gpu_config.USE_GPU` is ``True`` (OpenCL
      available), the 804×804 work buffer is uploaded to the GPU as a
      :class:`cv2.UMat` so that ``cv2.threshold``, ``cv2.erode`` ×5, and
      ``cv2.GaussianBlur`` execute on the OpenCL device.  The buffer is
      downloaded back to a NumPy array only for the final ``cv2.resize``
      and the NumPy slice assignment.  On a stock Raspberry Pi 4 (no
      functional OpenCL driver) ``USE_GPU`` is ``False`` and the path is
      identical to the original CPU-only implementation.

    Parameters
    ----------
    image:
        BGR source image (typically the output of :func:`extract_circles`)
        with shape ``(H, W, 3)`` and dtype ``uint8``.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` image of the same spatial dimensions as
        *image*, containing the blurred binary mask suitable for passing
        to :func:`cv2.findContours`.  Pixels outside the ROI bounding-box
        are zero.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Crop to the bounding-box of the circular ROI (~2.6 MP vs ~5 MP).
    roi_gray = gray[_ROI_Y0:_ROI_Y1, _ROI_X0:_ROI_X1]

    # Downscale 2× for cheaper morphological operations (804×804 px).
    roi_half = cv2.resize(
        roi_gray, (_ROI_HALF_W, _ROI_HALF_H), interpolation=cv2.INTER_AREA
    )

    # Optionally upload to the OpenCL device for the heavy operations.
    # cv2.UMat transparently routes threshold / erode / GaussianBlur through
    # OpenCL when USE_GPU is True; falls back to CPU automatically otherwise.
    work = cv2.UMat(roi_half) if USE_GPU else roi_half

    effective_threshold = int(np.clip(threshold_value, 0, 255))
    _ret, work = cv2.threshold(work, effective_threshold, 255, cv2.THRESH_BINARY)
    work = cv2.erode(work, _EROSION_KERNEL_HALF, iterations=_EROSION_ITERATIONS)
    # 11×11 is the half-scale equivalent of the original 21×21 kernel.
    work = cv2.GaussianBlur(work, (11, 11), 0)

    # Upscale back to full ROI dimensions.  INTER_LINEAR is intentional: the
    # image is not purely binary at this point (GaussianBlur produced smooth
    # gradient edges), so a bilinear upscale preserves that gradient more
    # faithfully than INTER_NEAREST and yields cleaner contours for fitEllipse.
    work = cv2.resize(work, (_ROI_W, _ROI_H), interpolation=cv2.INTER_LINEAR)

    # Download from the OpenCL device back to a NumPy array before the
    # slice assignment below (UMat cannot be used as an rvalue in ndarray
    # fancy indexing).
    threshed_roi: np.ndarray = work.get() if USE_GPU else work  # type: ignore[union-attr]

    # Embed the ROI result in a full-size output to preserve the public
    # interface expected by ellipse_fit and image_processing.py.
    result = np.zeros(image.shape[:2], dtype=np.uint8)
    result[_ROI_Y0:_ROI_Y1, _ROI_X0:_ROI_X1] = threshed_roi
    return result


def ellipse_fit(image: np.ndarray, cont_img: np.ndarray) -> np.ndarray:
    """Fit an ellipse to the largest contour and mask the source image.

    Finds all external contours in *cont_img*, selects the one with the
    greatest area, fits an ellipse to it, then applies the resulting filled
    ellipse as a single-channel mask to *image*.

    Performance (Raspberry Pi): same single-channel mask + bitwise_and
    strategy as :func:`extract_circles`.

    Parameters
    ----------
    image:
        BGR source image (typically the output of :func:`extract_circles`)
        with shape ``(H, W, 3)`` and dtype ``uint8``.
    cont_img:
        Single-channel binary image (output of :func:`erode_thresh`) used
        for contour detection.

    Returns
    -------
    np.ndarray
        BGR image of the same shape and dtype as *image* with all pixels
        outside the fitted ellipse zeroed out.

    Raises
    ------
    ValueError
        If *cont_img* contains no detectable contours.
    """
    contours, _hierarchy = cv2.findContours(
        cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No contours found in the thresholded image.")

    largest_contour = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(largest_contour)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)

    return cv2.bitwise_and(image, image, mask=mask)


def extract_fundus_from_image(
    image: np.ndarray,
    threshold_value: int = _THRESHOLD_VALUE,
) -> np.ndarray:
    """Return the extracted fundus region from an already-loaded image."""
    circle = extract_circles(image)
    threshed_image = erode_thresh(circle, threshold_value=threshold_value)
    return ellipse_fit(circle, threshed_image)


def extract_fundus(filename: str, threshold_value: int = _THRESHOLD_VALUE) -> np.ndarray:
    """Load an image file and return the extracted fundus region.

    Convenience wrapper that chains :func:`extract_circles`,
    :func:`erode_thresh`, and :func:`ellipse_fit` into a single call
    suitable for use in the processing pipeline.

    Parameters
    ----------
    filename:
        Absolute or relative path to the source JPEG/PNG image captured
        by the OWL fundus camera.

    Returns
    -------
    np.ndarray
        BGR image containing only the retinal disc region; background
        pixels are black.

    Raises
    ------
    FileNotFoundError
        If *filename* cannot be read by OpenCV.
    ValueError
        If no fundus contour can be detected in the image.
    """
    test_img = cv2.imread(filename)
    if test_img is None:
        raise FileNotFoundError(f"Cannot read image file: {filename!r}")

    return extract_fundus_from_image(test_img, threshold_value=threshold_value)
