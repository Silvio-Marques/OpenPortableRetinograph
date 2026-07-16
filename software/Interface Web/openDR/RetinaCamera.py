"""
Camera lifecycle management with contrast enhancement for the openDR pipeline.

Provides :class:`RetinaCamera`, a replacement for :class:`Fundus_Cam` that adds:

* Context-manager support so the hardware is always released cleanly.
* Distinct exception types for the three most common Pi camera failure modes:
  interface errors, overheating, and cable/module disconnection.
* A real-time CLAHE contrast-enhancement pass applied to every frame before
  JPEG encoding, improving retinal-vessel visibility for downstream grading.
"""
from __future__ import annotations

import logging
import time
import types
from pathlib import Path
from threading import Thread
from typing import Final

import cv2
import numpy as np
from picamera2 import Picamera2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLAHE parameters (tuned for fundus imagery)
# ---------------------------------------------------------------------------

# Clip limit for CLAHE – higher values increase contrast but amplify noise.
_CLAHE_CLIP_LIMIT: Final[float] = 2.0

# Tile grid size for CLAHE (width, height in pixels).
_CLAHE_TILE_GRID: Final[tuple[int, int]] = (8, 8)

# JPEG quality passed to cv2.imencode.
_JPEG_QUALITY: Final[int] = 95

# Maximum number of start-up retries before raising CameraInterfaceError.
_MAX_START_RETRIES: Final[int] = 3

# Seconds to wait between start-up retries.
_RETRY_DELAY_S: Final[float] = 1.0


# ---------------------------------------------------------------------------
# Public exception hierarchy
# ---------------------------------------------------------------------------


class RetinaCameraError(RuntimeError):
    """Base class for all :class:`RetinaCamera` hardware errors."""


class CameraInterfaceError(RetinaCameraError):
    """Raised when the camera interface cannot be initialised or configured.

    Typical causes include a missing kernel module, a corrupt device node,
    or a bad ``libcamera`` / Picamera2 installation.
    """


class CameraOverheatError(RetinaCameraError):
    """Raised when the camera module reports a thermal fault.

    On Raspberry Pi, sustained high CPU or ambient temperature can cause
    the ISP to return error frames or to stop the camera mid-session.
    """


class CameraDisconnectedError(RetinaCameraError):
    """Raised when the camera is no longer reachable during a capture.

    This covers physical disconnection (ribbon cable, CSI connector) as
    well as runtime errors that indicate the device has gone away.
    """


# ---------------------------------------------------------------------------
# RetinaCamera
# ---------------------------------------------------------------------------


class RetinaCamera:
    """Picamera2-backed camera with lifecycle management and contrast enhancement.

    Parameters
    ----------
    framerate:
        Target frame rate passed to Picamera2's still configuration.
    max_frames:
        Maximum number of frames stored during a :meth:`continuous_capture`
        run before the background thread stops automatically.
    jpeg_quality:
        JPEG encoding quality (1–100) used by :meth:`capture` and
        :meth:`capture_to_file`.

    Examples
    --------
    Use as a context manager to guarantee clean hardware teardown::

        with RetinaCamera() as cam:
            data = cam.capture()

    Or manage the lifecycle explicitly::

        cam = RetinaCamera()
        cam.start()
        try:
            data = cam.capture()
        finally:
            cam.stop()

    Raises
    ------
    CameraInterfaceError
        If the Picamera2 device cannot be opened or configured after
        :data:`_MAX_START_RETRIES` attempts.
    """

    def __init__(
        self,
        framerate: int = 12,
        max_frames: int = 10,
        jpeg_quality: int = _JPEG_QUALITY,
    ) -> None:
        self._framerate = framerate
        self._max_frames = max_frames
        self._jpeg_quality = jpeg_quality

        self._camera: Picamera2 | None = None
        self._flip_state: bool = False
        self._stopped: bool = False
        self._capture_thread: Thread | None = None

        self.images: list[np.ndarray] = []
        self.image: np.ndarray | None = None

        self._clahe = cv2.createCLAHE(
            clipLimit=_CLAHE_CLIP_LIMIT,
            tileGridSize=_CLAHE_TILE_GRID,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "RetinaCamera":
        """Initialise and start the camera hardware.

        Attempts to open and configure Picamera2 up to
        :data:`_MAX_START_RETRIES` times, sleeping :data:`_RETRY_DELAY_S`
        seconds between attempts.

        Returns
        -------
        RetinaCamera
            *self*, so the call can be chained: ``cam = RetinaCamera().start()``.

        Raises
        ------
        CameraInterfaceError
            If the camera cannot be opened after all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_START_RETRIES + 1):
            try:
                camera = Picamera2()
                sensor_resolution = camera.sensor_resolution
                config = camera.create_still_configuration(
                    main={"size": sensor_resolution},
                    controls={"FrameRate": self._framerate},
                )
                camera.configure(config)
                camera.start()
                self._camera = camera
                logger.info("RetinaCamera started (attempt %d/%d).", attempt, _MAX_START_RETRIES)
                return self
            except Exception as exc:  # noqa: BLE001 – hardware init can raise anything
                last_exc = exc
                logger.warning(
                    "Camera start failed (attempt %d/%d): %s",
                    attempt,
                    _MAX_START_RETRIES,
                    exc,
                )
                if attempt < _MAX_START_RETRIES:
                    time.sleep(_RETRY_DELAY_S)

        raise CameraInterfaceError(
            f"Camera could not be initialised after {_MAX_START_RETRIES} attempts."
        ) from last_exc

    def stop(self) -> None:
        """Stop the camera hardware and release all resources."""
        self._stopped = True
        if self._camera is not None and self._camera.started:
            try:
                self._camera.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while stopping camera: %s", exc)
            finally:
                self._camera = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RetinaCamera":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def flip_cam(self) -> None:
        """Toggle vertical flip on subsequent captures."""
        self._flip_state = not self._flip_state

    # ------------------------------------------------------------------
    # Internal capture helpers
    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        """Raise :exc:`CameraDisconnectedError` if the camera is not active."""
        if self._camera is None or not self._camera.started:
            raise CameraDisconnectedError(
                "Camera is not running. Call start() before capturing."
            )

    def _apply_contrast_enhancement(self, bgr: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement to a BGR frame.

        The image is converted to LAB colour space so that CLAHE is applied
        only to the lightness channel (L), preserving hue and saturation.
        The enhanced L channel is merged back with the original A and B
        channels before conversion to BGR.

        Parameters
        ----------
        bgr:
            Source frame as a ``(H, W, 3)`` ``uint8`` BGR array.

        Returns
        -------
        np.ndarray
            Contrast-enhanced BGR array with the same shape and dtype.
        """
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_ch)
        lab_enhanced = cv2.merge((l_enhanced, a_ch, b_ch))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _read_raw_frame(self) -> np.ndarray:
        """Grab one raw BGR frame from the sensor, applying flip and fault checks.

        Shared by :meth:`_capture_jpeg_buffer` and :meth:`capture_preview` so
        disconnect/overheat detection lives in exactly one place.

        Raises
        ------
        CameraDisconnectedError
            If the camera stops responding during the capture.
        CameraOverheatError
            If a thermal fault is detected (heuristic: ISP returns an
            all-zero frame, which is the observed behaviour on overheating
            Pi hardware).
        """
        self._assert_ready()

        try:
            frame = self._camera.capture_array()
        except Exception as exc:
            # Picamera2 does not expose a public thermal-fault exception type.
            # Inspect the message as a best-effort heuristic; callers can
            # catch RetinaCameraError to handle both failure modes uniformly.
            msg = str(exc).lower()
            if "temperature" in msg or "thermal" in msg or "overheat" in msg:
                raise CameraOverheatError(
                    "Camera thermal fault detected during capture."
                ) from exc
            raise CameraDisconnectedError(
                f"Camera disconnected or unresponsive during capture: {exc}"
            ) from exc

        # Heuristic: an all-zero frame can indicate an ISP/thermal failure.
        # frame.max() short-circuits as soon as a non-zero value is found,
        # making it faster than np.any() on high-resolution sensor arrays.
        if frame.max() == 0:
            raise CameraOverheatError(
                "Captured frame is entirely black – possible thermal fault or "
                "camera hardware failure."
            )

        if self._flip_state:
            frame = cv2.flip(frame, 0)

        return frame

    def _capture_jpeg_buffer(self) -> np.ndarray:
        """Capture a single frame, enhance contrast, and return a JPEG buffer.

        Returns
        -------
        np.ndarray
            1-D ``uint8`` array containing the raw JPEG bytes.

        Raises
        ------
        CameraDisconnectedError
            If the camera stops responding during the capture.
        CameraOverheatError
            If a thermal fault is detected.
        RuntimeError
            If OpenCV cannot encode the frame as a JPEG.
        """
        frame = self._read_raw_frame()
        enhanced = self._apply_contrast_enhancement(frame)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        encode_success, image_bytes = cv2.imencode(".jpg", enhanced, encode_params)
        if not encode_success:
            raise RuntimeError(
                "Unable to encode camera frame as JPEG. "
                "Check camera connection and frame data."
            )

        return np.frombuffer(image_bytes.tobytes(), dtype=np.uint8)

    def capture_preview(
        self, max_dimension: int = 640, jpeg_quality: int = 80
    ) -> np.ndarray:
        """Capture a reduced-size, contrast-enhanced preview frame as JPEG bytes.

        The frame is downscaled *before* CLAHE is applied, so the enhancement
        pass runs on the small preview-sized array rather than the full
        sensor resolution — this keeps the endpoint safe to poll at
        interactive rates (see ``PREVIEW_MIN_INTERVAL_S`` in ``fundus.py``).

        Parameters
        ----------
        max_dimension:
            Maximum width or height for the preview image.
        jpeg_quality:
            JPEG quality used during encoding (0-100).

        Returns
        -------
        np.ndarray
            1-D ``uint8`` array containing the encoded JPEG bytes.

        Raises
        ------
        CameraDisconnectedError
            If the camera is not running or disconnects during capture.
        CameraOverheatError
            If a thermal fault is detected.
        RuntimeError
            If OpenCV cannot encode the frame as a JPEG.
        """
        frame = self._read_raw_frame()

        height, width = frame.shape[:2]
        largest_dimension = max(height, width)
        if largest_dimension > max_dimension:
            scale = max_dimension / float(largest_dimension)
            frame = cv2.resize(
                frame,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        enhanced = self._apply_contrast_enhancement(frame)

        encode_success, image_bytes = cv2.imencode(
            ".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if not encode_success:
            raise RuntimeError(
                "Unable to encode camera preview frame as JPEG. "
                "Check camera connection and frame data."
            )

        return np.frombuffer(image_bytes.tobytes(), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Public capture API
    # ------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """Capture a single contrast-enhanced JPEG frame.

        Returns
        -------
        np.ndarray
            1-D ``uint8`` array containing the raw JPEG bytes, also stored
            in :attr:`image`.

        Raises
        ------
        CameraDisconnectedError
            If the camera is not running or disconnects during capture.
        CameraOverheatError
            If a thermal fault is detected.
        RuntimeError
            If JPEG encoding fails.
        """
        self.image = self._capture_jpeg_buffer()
        return self.image

    def capture_to_file(self, path: str | Path) -> Path:
        """Capture a single frame and save it directly to *path* as a JPEG.

        Parameters
        ----------
        path:
            Destination file path. The ``.jpg`` extension is recommended;
            the file is written in binary mode regardless of the extension.

        Returns
        -------
        Path
            The resolved output path.

        Raises
        ------
        CameraDisconnectedError
            If the camera is not running or disconnects during capture.
        CameraOverheatError
            If a thermal fault is detected.
        RuntimeError
            If JPEG encoding fails.
        OSError
            If the file cannot be written to *path*.
        """
        jpeg_buffer = self._capture_jpeg_buffer()
        dest = Path(path).resolve()
        dest.write_bytes(jpeg_buffer.tobytes())
        logger.info("Frame saved to %s (%d bytes).", dest, len(jpeg_buffer))
        return dest

    def continuous_capture(self) -> None:
        """Begin capturing frames in a background thread.

        Captured JPEG buffers are appended to :attr:`images`.  The thread
        stops automatically once :attr:`max_frames` frames are collected or
        a hardware error occurs (errors are logged but do not propagate to
        the calling thread).

        Call :meth:`wait_for_capture` to block until the thread finishes.
        """
        self._stopped = False
        self.images = []
        self._capture_thread = Thread(target=self._update, daemon=True)
        self._capture_thread.start()

    def _update(self) -> None:
        """Background thread target for :meth:`continuous_capture`."""
        while not self._stopped:
            try:
                self.images.append(self._capture_jpeg_buffer())
            except RetinaCameraError as exc:
                logger.error("Hardware error during continuous capture: %s", exc)
                self._stopped = True
                return
            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected error during continuous capture: %s", exc)
                self._stopped = True
                return

            if len(self.images) >= self._max_frames:
                self._stopped = True
                return

    def wait_for_capture(self, timeout: float = 5.0) -> bool:
        """Block until the background capture thread finishes.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.

        Returns
        -------
        bool
            ``True`` if the thread finished within *timeout*, ``False``
            otherwise.
        """
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=timeout)
        return self._stopped
