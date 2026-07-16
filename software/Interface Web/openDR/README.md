## Open Indirect Ophthalmoscope (OIO / OWL)
_Original authors: Ayush Yadav, Ebin Philip, Dhruv Joshi_  
_Maintained by: [@heliobentzen](https://github.com/heliobentzen), GitHub Copilot_

Built at the Srujana Center for Innovation, LV Prasad Eye Institute, Hyderabad, India.

## LICENSE AND DISCLAIMER
This software is covered by the MIT License (see `license.txt`).

## Version
Current release: **4.0.1**

## What's New in 4.0.1
- **Guided Grad-CAM fix** — in-place ReLU operations are temporarily disabled during guided backpropagation to prevent backward-hook inplace-modification errors; gradient masking now correctly follows Springenberg et al. (2015).
- **Adaptive lesion thresholding** — lesion detection uses Otsu's method with a configurable safety floor instead of a fixed threshold, improving robustness across varying exposures.
- **Watershed lesion splitting** — touching lesion blobs are separated into distinct regions using a distance-transform watershed, reducing false merges.
- **Morphological denoising** — open/close passes remove small artefacts from the binary activation mask before connected-component analysis.
- **Richer lesion metrics** — each detected region now reports `circularity` (0–1 shape roundness) and `relative_intensity` (activation relative to whole-image mean).
- **Audit JSON schema v1.2** — both overlay paths (`_gradcam.jpg` and `_guided_gradcam.jpg`) and the new metrics are included in the JSON audit record.

## What's New in 4.0.0
- **RetinaCamera class** — dedicated camera abstraction with lifecycle management, hardware error handling, and CLAHE contrast enhancement for improved image quality.
- **Grad-CAM explainability** — visual saliency maps overlaid on retinal images to highlight regions driving the DR classification decision.
- **Live inference progress** — real-time grading status streamed to the UI via background executor, with clear error and completion feedback.
- **Client-side focus gating** — the capture button is only enabled after the preview frame passes a sharpness threshold; the backend re-validates focus before saving the image.
- **Picamera2 preview endpoint** — `/preview-frame` streams JPEG frames from the libcamera stack for low-latency live preview.
- **Tailwind CSS UI** — all HTML templates rebuilt with Tailwind CSS for a modern, responsive interface.
- **Modernised screens** — start screen and action buttons redesigned with consistent icon-based layout.
- **Security hardening** — sanitised error messages, strengthened `serve_image` path validation to prevent directory traversal.
- **Python 3.11 type hints & docstrings** — processing and module files fully annotated for maintainability.

## Supported Platform
- Raspberry Pi 4
- Latest Raspberry Pi OS (Bullseye/Bookworm, 64-bit recommended)
- libcamera stack (`Picamera2`) for camera access

## Runtime Dependencies
- Python 3.11+
- OpenCV 4.x (`python3-opencv`)
- Picamera2 / libcamera (`python3-picamera2`, `libcamera-apps`)
- Flask, NumPy, Requests, pigpio

## Installation
Clone into `/home/pi/openDR` on the Raspberry Pi:

```bash
cd /home/pi
git clone <repo-url> openDR
cd /home/pi/openDR
bash install.sh
```

## Running the application
```bash
cd /home/pi/openDR
python3 fundus.py
```

## Notes on camera migration
- Legacy `picamera` usage has been replaced with `Picamera2` (libcamera backend).
- Vertical camera flipping is now handled in software before JPEG encoding.

## Folder structure
`images` contains captured patient/session images (patient ID included in filenames).  
`modules` contains image processing and grading modules.  
`static` and `templates` provide Flask UI assets.
