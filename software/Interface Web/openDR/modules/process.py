"""
End-to-end image processing pipeline for the openDR fundus grading system.

This module chains fundus extraction (circle masking + ellipse fitting) with
LED-glare removal and forwards the result to the Theia grading API.  An
optional Grad-CAM explanation step is available via
:func:`grade_with_explanation`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from .extract import extract_fundus_from_image
from . import gradcam, remove_glare, theia


DEFAULT_PROCESSING_SETTINGS = {
    "brightness": 0,
    "contrast": 100,
    "fundus_threshold": 65,
    "glare_threshold": 90,
}


def _coerce_setting(
    value: Any,
    fallback: int,
) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def normalize_processing_settings(
    processing_settings: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    settings = dict(DEFAULT_PROCESSING_SETTINGS)
    if processing_settings is not None:
        settings.update(processing_settings)

    return {
        "brightness": int(
            np.clip(_coerce_setting(settings["brightness"], 0), -100, 100)
        ),
        "contrast": int(
            np.clip(_coerce_setting(settings["contrast"], 100), 50, 180)
        ),
        "fundus_threshold": int(
            np.clip(_coerce_setting(settings["fundus_threshold"], 65), 0, 255)
        ),
        "glare_threshold": int(
            np.clip(_coerce_setting(settings["glare_threshold"], 90), 50, 100)
        ),
    }


def apply_processing_settings(
    image: np.ndarray,
    processing_settings: Mapping[str, Any] | None = None,
) -> np.ndarray:
    settings = normalize_processing_settings(processing_settings)
    adjusted = image.astype(np.float32)
    adjusted = (adjusted * (settings["contrast"] / 100.0)) + settings["brightness"]
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def prepare_processed_image(
    filename: str,
    processing_settings: Mapping[str, Any] | None = None,
) -> np.ndarray:
    source_image = cv2.imread(filename)
    if source_image is None:
        raise FileNotFoundError(f"Cannot read image file: {filename!r}")

    settings = normalize_processing_settings(processing_settings)
    adjusted = apply_processing_settings(source_image, settings)
    extracted = extract_fundus_from_image(
        adjusted,
        threshold_value=settings["fundus_threshold"],
    )
    return remove_glare.remove_glare(
        extracted,
        saturation_threshold=settings["glare_threshold"] / 100.0,
    )


def grade(filename: str, processing_settings: Mapping[str, Any] | None = None) -> float:
    """Process a raw fundus image and return its diabetic-retinopathy grade.

    The pipeline consists of three steps:

    1. **Extraction** – :func:`~modules.extract.extract_fundus` isolates the
       retinal disc by applying a circular crop followed by an ellipse-fitted
       mask.
    2. **Glare removal** – :func:`~modules.remove_glare.remove_glare`
       inpaints the two direct LED specular reflections from the 20D lens.
    3. **Grading** – :func:`~modules.theia.grade_request` uploads the
       processed image to the Theia API and returns a numeric DR grade.

    The processed image is written to disk as a JPEG (``<stem>_processed.jpg``
    in the same directory as *filename*) so it can be inspected manually
    and sent to the API by file path.

    Parameters
    ----------
    filename:
        Path to the raw fundus JPEG captured by the OWL camera.

    Returns
    -------
    float
        Diabetic-retinopathy grade returned by the Theia API, or ``-1``
        if the API call fails.
    """
    source_path = Path(filename)
    output = str(source_path.with_name(source_path.stem + "_processed.jpg"))
    cv2.imwrite(output, prepare_processed_image(filename, processing_settings))
    return theia.grade_request(output)


def grade_with_explanation(
    filename: str,
    model_path: str | None = None,
    status_callback: Callable[..., None] | None = None,
    processing_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Process a fundus image, grade it, and generate a Grad-CAM explanation.

    Runs the same extraction / glare-removal pipeline as :func:`grade`, then
    applies Grad-CAM to produce a colour-coded heatmap overlay and a JSON
    audit record that documents the critical lesion regions detected.

    Output files written alongside *filename*:

    * ``<stem>_processed.jpg`` – glare-corrected fundus image.
    * ``<stem>_gradcam.jpg`` – heatmap overlay highlighting lesion regions.
    * ``<stem>_gradcam.json`` – structured audit record (DR grade, confidence,
      bounding-box coordinates of detected lesions).

    Parameters
    ----------
    filename:
        Path to the raw fundus JPEG captured by the OWL camera.
    model_path:
        Optional path to a trained ``.pt`` / ``.pth`` checkpoint used by
        Grad-CAM.  Falls back to the ``OPEN_DR_MODEL_PATH`` environment
        variable; when neither is set the module operates in demo mode with
        random weights.

    Returns
    -------
    dict
        A mapping with keys:

        ``theia_grade``
            Numeric DR grade returned by the Theia API (``-1`` on failure).
        ``gradcam``
            The full Grad-CAM audit record (see
            :func:`~modules.gradcam.run_gradcam`).

    Raises
    ------
    RuntimeError
        If PyTorch / torchvision is not installed (raised by the Grad-CAM
        module).
    """
    source_path = Path(filename)
    processed_path = str(source_path.with_name(source_path.stem + "_processed.jpg"))

    processed_image = prepare_processed_image(filename, processing_settings)
    cv2.imwrite(processed_path, processed_image)
    if status_callback is not None:
        status_callback("preprocessing", processed_path=processed_path)

    theia_grade = theia.grade_request(processed_path)
    if status_callback is not None:
        status_callback(
            "inference",
            processed_path=processed_path,
            theia_grade=theia_grade,
        )
    try:
        gradcam_record = gradcam.run_gradcam(
            processed_image, processed_path, model_path=model_path
        )
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Grad-CAM explanation step failed: {exc}") from exc

    if status_callback is not None:
        status_callback(
            "report",
            processed_path=processed_path,
            theia_grade=theia_grade,
            gradcam=gradcam_record,
        )

    return {
        "theia_grade": theia_grade,
        "gradcam": gradcam_record,
    }
