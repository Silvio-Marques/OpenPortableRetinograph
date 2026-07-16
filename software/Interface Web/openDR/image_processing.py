"""
Interactive demo script for the openDR fundus image-extraction pipeline.

Displays each intermediate result in a resized window so the output of
:func:`~modules.extract.extract_circles`, :func:`~modules.extract.erode_thresh`,
and :func:`~modules.extract.ellipse_fit` can be inspected visually without
running the full grading pipeline.

Run directly::

    python image_processing.py [path/to/image.jpg]

Requires the ``imutils`` package (``pip install imutils``).
"""
from __future__ import annotations

import cv2

from modules.extract import ellipse_fit, erode_thresh, extract_circles


def _load_imutils():
    try:
        import imutils as imported_imutils
    except ImportError:  # pragma: no cover - only needed for standalone demo execution
        return None
    return imported_imutils


imutils = _load_imutils()


def main(image_path: str = "owl1.jpg") -> None:
    """Run the extraction pipeline and display each stage in a window.

    Parameters
    ----------
    image_path:
        Path to a fundus JPEG captured by the OWL camera.

    Raises
    ------
    ImportError
        If the ``imutils`` package is not installed.
    FileNotFoundError
        If *image_path* cannot be read by OpenCV.
    """
    if imutils is None:
        raise ImportError("imutils is required to run image_processing.py")

    test_img = cv2.imread(image_path)
    if test_img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    circle = extract_circles(test_img)
    cv2.imshow("extracted circle", imutils.resize(circle, width=432, height=324))

    threshed_image = erode_thresh(circle)
    cv2.imshow(
        "eroded and threshed",
        imutils.resize(threshed_image, width=432, height=324),
    )

    final_image = ellipse_fit(circle, threshed_image)
    cv2.imshow("window", imutils.resize(final_image, width=432, height=324))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

