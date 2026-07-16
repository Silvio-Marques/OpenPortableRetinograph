"""
Grad-CAM and Guided Grad-CAM for diabetic-retinopathy classification on fundus
images.

Implements:

* **Grad-CAM** – Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
  Networks via Gradient-based Localization", ICCV 2017.
* **Guided Backpropagation** – Springenberg et al., "Striving for Simplicity:
  The All Convolutional Net", ICLR 2015 workshop.
* **Guided Grad-CAM** – element-wise product of the upsampled Grad-CAM map and
  the guided-backpropagation gradient, combining class-discriminative
  localisation with pixel-precise edge detail.

The module:

* Runs a ResNet-18 classification model (fine-tuned or demo) on a processed
  fundus image.
* Extracts gradients from the final convolutional block to build the
  class-discriminative activation map (CAM) – ``<stem>_gradcam.jpg``.
* Computes guided backpropagation by hooking every ReLU in the network so that
  only positive gradients flowing through positive activations are propagated,
  yielding a pixel-level saliency map.
* Multiplies the upsampled Grad-CAM with the guided-backprop magnitude to
  produce **Guided Grad-CAM** – ``<stem>_guided_gradcam.jpg`` – which highlights
  the exact edges of small retinal lesions instead of coarse circular blobs.
* Identifies high-activation lesion regions (microaneurysms, exudates,
  haemorrhages) and writes their bounding-box coordinates together with the
  full audit record to ``<stem>_gradcam.json``.

Environment variables
---------------------
OPEN_DR_MODEL_PATH
    Path to a ``.pt`` / ``.pth`` checkpoint.  When absent or the file does
    not exist the module operates in *demo mode* with random weights – useful
    for integration testing without a trained checkpoint.

Performance notes (Raspberry Pi 4)
------------------------------------
Several optimisations are applied to reduce per-call latency on the Pi 4:

* **Model cache** – loaded ``nn.Module`` instances are stored in the module-
  level ``_model_cache`` dict, keyed by ``(resolved_path, num_classes)``.
  Subsequent calls with the same checkpoint avoid the ~0.5–1 s file-I/O and
  weight-copy overhead.  Restart the service to pick up a new checkpoint.

* **Pre-built transform** – the torchvision ``Compose`` pipeline is
  constructed once (``_PREPROCESS_TRANSFORM``) rather than inside every
  ``_preprocess`` call.

* **Pre-allocated morphological kernel** – ``_MORPH_KERNEL_3x3`` uses
  ``cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))``, matching the
  pattern in ``extract.py`` / ``remove_glare.py`` and enabling OpenCV's
  separable-decomposition fast path.

* **CPU thread hint** – ``torch.set_num_threads`` is called once at import
  time with ``os.cpu_count()`` so PyTorch's BLAS backend uses all available
  Cortex-A72 cores.

**Why TFLite / quantization is not used here**
Both Grad-CAM and Guided Grad-CAM require a live autograd backward pass to
extract gradients.  TFLite and ``torch.quantization.quantize_dynamic`` do
not support gradient flow, so they cannot be applied to the model used for
explanation without silently producing incorrect heatmaps.  Quantization
could be applied to a *separate* inference-only copy to obtain
``pred_class`` / ``confidence`` faster, but that adds a third forward pass
and extra RAM – a net loss on the Pi.  This decoupling is left as future
work for when the Theia grading inference and the Grad-CAM explanation step
are run on separate hardware.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    # Use all available CPU cores for BLAS operations.  On Pi 4 (4 × Cortex-A72)
    # this can halve matrix-multiply time; it is a no-op on hardware where
    # PyTorch already auto-detects the core count.
    torch.set_num_threads(os.cpu_count() or 1)

    #: ImageNet pre-processing pipeline built once and reused across all calls.
    _PREPROCESS_TRANSFORM = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: ICDR (International Clinical Diabetic Retinopathy) scale labels.
DR_CLASSES: list[str] = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "PDR",
]

#: Fallback activation fraction above which a pixel is treated as a potential
#: lesion site when adaptive thresholding would be too permissive.
_LESION_THRESHOLD: float = 0.5

#: Minimum bounding-box area (original-image pixels) required to report a
#: lesion region (guards against noise artefacts).
_MIN_LESION_AREA: int = 100

#: Fraction of the distance-transform peak used to split touching lesion blobs
#: into separate connected components.
_LESION_SPLIT_RATIO: float = 0.35

#: Opacity of the heatmap overlay blended onto the original image.
_OVERLAY_ALPHA: float = 0.45

#: Pre-allocated 3×3 rectangular structuring element shared by all
#: morphological operations in ``_extract_lesion_regions``.  Uses
#: ``MORPH_RECT`` so OpenCV can apply its separable row+column fast path,
#: matching the pattern in ``extract.py`` and ``remove_glare.py``.
_MORPH_KERNEL_3x3: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

#: In-process model cache: maps ``(resolved_model_path, num_classes)`` to the
#: already-loaded ``nn.Module``.  Avoids repeated file I/O and weight-copy
#: on every ``run_gradcam`` call.  Restart the service to pick up a new
#: checkpoint.
_model_cache: "dict[tuple[str | None, int], nn.Module]" = {}


# ---------------------------------------------------------------------------
# Internal hook helper
# ---------------------------------------------------------------------------


class _GradCAMHook:
    """Capture forward activations and backward gradients from a layer."""

    def __init__(self) -> None:
        self.activations: "torch.Tensor | None" = None
        self.gradients: "torch.Tensor | None" = None

    def forward_hook(
        self,
        _module: "nn.Module",
        _input: tuple,
        output: "torch.Tensor",
    ) -> None:
        self.activations = output.detach()

    def backward_hook(
        self,
        _module: "nn.Module",
        _grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        self.gradients = grad_output[0].detach()


# ---------------------------------------------------------------------------
# Model construction and loading
# ---------------------------------------------------------------------------


def _build_model(num_classes: int = 5) -> "nn.Module":
    """Return a ResNet-18 with a custom head for *num_classes* DR grades."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _load_model(
    model_path: str | None = None,
    num_classes: int = 5,
    device: "torch.device | None" = None,
) -> "nn.Module":
    """Load (or initialise) the classification model.

    If *model_path* points to an existing file the saved weights are
    loaded; otherwise the model is returned with random weights
    (demo / integration-test mode).

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` / ``.pth`` checkpoint.  Bare ``state_dict``
        files and checkpoints wrapped in a ``{"model_state_dict": …}``
        mapping are both supported.
    num_classes:
        Number of output classes (default: 5 for the ICDR scale).
    device:
        Target device.  Defaults to CPU.
    """
    if device is None:
        device = torch.device("cpu")

    # Resolve the path to a canonical string so that relative and absolute
    # references to the same file share one cache entry.
    resolved = str(Path(model_path).resolve()) if model_path else None
    cache_key = (resolved, num_classes)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    model = _build_model(num_classes=num_classes)

    if resolved and Path(resolved).is_file():
        state = torch.load(resolved, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    _model_cache[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------


def _preprocess(image: np.ndarray) -> "torch.Tensor":
    """Convert a BGR OpenCV image to a normalised 4-D tensor.

    Applies the standard ImageNet normalisation so the model weights are
    compatible with torchvision pre-trained initialisations.

    Parameters
    ----------
    image:
        BGR image as a ``uint8`` NumPy array.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(1, 3, 224, 224)``.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return _PREPROCESS_TRANSFORM(rgb).unsqueeze(0)  # (1, C, H, W)


# ---------------------------------------------------------------------------
# Core Grad-CAM computation
# ---------------------------------------------------------------------------


def _compute_gradcam(
    model: "nn.Module",
    tensor: "torch.Tensor",
    target_class: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """Compute the Grad-CAM activation map for *tensor*.

    Registers forward/backward hooks on the last residual block of the
    ResNet-18 backbone (``model.layer4[-1]``), performs a forward pass,
    back-propagates the score of *target_class*, then weights the captured
    feature maps by the global-average-pooled gradients.

    Parameters
    ----------
    model:
        Classification model (ResNet-18 or any model that exposes a
        ``layer4`` attribute containing residual blocks).
    tensor:
        Pre-processed input tensor of shape ``(1, 3, H, W)``.
    target_class:
        Class index for which to explain the prediction.  When ``None``
        the model's top-1 prediction is used.

    Returns
    -------
    tuple
        ``(cam, pred_class, confidence)`` where *cam* is a ``float32``
        array with values in ``[0, 1]`` at the spatial resolution of the
        last convolutional feature map, *pred_class* is the predicted
        class index, and *confidence* is the corresponding softmax score.
    """
    hook = _GradCAMHook()
    target_layer: "nn.Module" = model.layer4[-1]  # type: ignore[index]
    fwd_handle = target_layer.register_forward_hook(hook.forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(hook.backward_hook)

    try:
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, target_class].item())

        model.zero_grad()
        logits[0, target_class].backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    # Global-average-pool the gradients to obtain neuron importance weights.
    weights = hook.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = (weights * hook.activations).sum(dim=1).squeeze(0)  # (H, W)
    cam = torch.relu(cam).cpu().numpy()

    # Normalise to [0, 1].
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam.astype(np.float32), target_class, confidence


# ---------------------------------------------------------------------------
# Guided Backpropagation
# ---------------------------------------------------------------------------


def _compute_guided_backprop(
    model: "nn.Module",
    tensor: "torch.Tensor",
    target_class: int,
) -> np.ndarray:
    """Compute guided-backpropagation gradients at the input image.

    Registers temporary forward and backward hooks on every ``nn.ReLU``
    module in the network so that only *positive* gradients flowing through
    *positively-activated* neurons are propagated (Springenberg et al.,
    2015).  All hooks are removed before returning.

    Parameters
    ----------
    model:
        Classification model in eval mode.
    tensor:
        Pre-processed input tensor of shape ``(1, 3, H, W)`` on the target
        device.  A detached clone with ``requires_grad=True`` is created
        internally so the original tensor is never modified.
    target_class:
        Class index whose score is backpropagated.

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(3, H, W)`` containing the raw guided
        gradients at the input.  Values are **not** normalised so that the
        caller can combine them with the Grad-CAM map before normalisation.
    """
    handles: list = []
    relu_outputs: dict[int, "torch.Tensor"] = {}

    # Temporarily disable in-place ReLU operations to avoid the backward hook
    # error: "Output 0 of BackwardHookFunctionBackward is a view and is being
    # modified inplace".  The original state is saved and restored afterwards.
    relu_inplace_states: dict[int, bool] = {}
    for i, module in enumerate(model.modules()):
        if isinstance(module, nn.ReLU):
            relu_inplace_states[i] = module.inplace
            module.inplace = False

    def make_forward_hook(idx: int):
        def hook(
            _module: "nn.Module",
            _inp: tuple,
            output: "torch.Tensor",
        ) -> None:
            # Save detached ReLU output so the backward hook can mask
            # non-positive activations.  Only nn.ReLU modules are hooked
            # (see loop below), so output > 0 iff input > 0.
            relu_outputs[idx] = output.detach()

        return hook

    def make_backward_hook(idx: int):
        def hook(
            _module: "nn.Module",
            _grad_in: tuple,
            grad_out: tuple,
        ) -> tuple:
            # Guided-backprop rule: zero out negative upstream gradients and
            # positions where the forward activation was non-positive.
            positive_upstream = torch.clamp(grad_out[0], min=0)
            positive_activation = (relu_outputs[idx] > 0).float()
            guided = positive_upstream * positive_activation
            return (guided,)

        return hook

    idx = 0
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(make_forward_hook(idx)))
            handles.append(
                module.register_full_backward_hook(make_backward_hook(idx))
            )
            idx += 1

    try:
        model.zero_grad()
        input_tensor = tensor.clone().requires_grad_(True)
        logits = model(input_tensor)
        model.zero_grad()
        logits[0, target_class].backward()
        if input_tensor.grad is None:
            return np.zeros(tensor.shape[1:], dtype=np.float32)  # (3, H, W)
        guided_grads = input_tensor.grad.squeeze(0).cpu().numpy()  # (3, H, W)
    finally:
        for h in handles:
            h.remove()
        # Restore original inplace state of all ReLU modules.
        for i, module in enumerate(model.modules()):
            if isinstance(module, nn.ReLU) and i in relu_inplace_states:
                module.inplace = relu_inplace_states[i]

    return guided_grads.astype(np.float32)


# ---------------------------------------------------------------------------
# Guided Grad-CAM combination
# ---------------------------------------------------------------------------


def _compute_guided_gradcam(
    cam: np.ndarray,
    guided_grads: np.ndarray,
) -> np.ndarray:
    """Combine Grad-CAM with guided-backpropagation gradients.

    Upsamples the coarse Grad-CAM map to the input-image resolution and
    multiplies it element-wise with the L2 magnitude of the guided
    gradients.  The result is a pixel-precise, class-discriminative saliency
    map that preserves both the class localisation of Grad-CAM and the
    edge-level detail of guided backpropagation.

    Parameters
    ----------
    cam:
        Normalised Grad-CAM map ``(H_cam, W_cam)`` with values in
        ``[0, 1]`` (output of :func:`_compute_gradcam`).
    guided_grads:
        Raw guided-backpropagation gradients ``(3, H_in, W_in)`` (output of
        :func:`_compute_guided_backprop`).

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(H_in, W_in)`` with values in
        ``[0, 1]``.
    """
    _, h_in, w_in = guided_grads.shape

    # Upsample coarse Grad-CAM to input resolution using bilinear interpolation
    # to produce a smooth, continuous heatmap.
    cam_upsampled = cv2.resize(cam, (w_in, h_in), interpolation=cv2.INTER_LINEAR)  # (H_in, W_in)

    # Per-pixel L2 magnitude across channels.
    guided_magnitude = np.linalg.norm(guided_grads, axis=0)  # (H_in, W_in)

    # Element-wise product: class-discriminative × pixel-precise.
    guided_gradcam = cam_upsampled * guided_magnitude

    # Normalise to [0, 1].
    g_min, g_max = float(guided_gradcam.min()), float(guided_gradcam.max())
    if g_max > g_min:
        guided_gradcam = (guided_gradcam - g_min) / (g_max - g_min)
    else:
        guided_gradcam = np.zeros_like(guided_gradcam)

    return guided_gradcam.astype(np.float32)


# ---------------------------------------------------------------------------
# Heatmap overlay
# ---------------------------------------------------------------------------


def _overlay_heatmap(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = _OVERLAY_ALPHA,
) -> np.ndarray:
    """Blend the Grad-CAM heatmap over the original BGR image.

    Parameters
    ----------
    image:
        Original BGR fundus image (``uint8``).
    cam:
        Grad-CAM activation map with values in ``[0, 1]`` at any spatial
        resolution – it is resized to match *image*.
    alpha:
        Opacity of the heatmap layer (0 = transparent, 1 = fully opaque).

    Returns
    -------
    np.ndarray
        BGR ``uint8`` composite image.
    """
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)


# ---------------------------------------------------------------------------
# Lesion region extraction
# ---------------------------------------------------------------------------


def _extract_lesion_regions(
    cam: np.ndarray,
    image_shape: tuple[int, int],
    threshold: float = _LESION_THRESHOLD,
    min_area: int = _MIN_LESION_AREA,
) -> list[dict[str, Any]]:
    """Detect high-activation lesion regions and return their coordinates.

    Applies adaptive thresholding (Otsu with a safety floor), uses connected
    components plus watershed splitting to separate touching lesion clusters,
    then computes bounding boxes and clinical region metrics scaled to the
    original image resolution.

    Parameters
    ----------
    cam:
        Normalised Grad-CAM map ``(H_cam, W_cam)`` with values in
        ``[0, 1]``.
    image_shape:
        ``(height, width)`` of the original image used to scale bounding
        boxes from the CAM resolution.
    threshold:
        Activation fraction above which a pixel is classified as a lesion
        candidate.
    min_area:
        Minimum bounding-box area (original pixels) to include a region.

    Returns
    -------
    list of dict
        Each entry contains the keys ``x``, ``y``, ``width``, ``height``,
        ``mean_activation``, ``relative_intensity``, ``circularity``, and
        ``area`` in original-image pixel coordinates.  The list is sorted by
        descending ``mean_activation`` (most significant lesions first).
    """
    orig_h, orig_w = image_shape
    cam_h, cam_w = cam.shape

    scale_x = orig_w / cam_w
    scale_y = orig_h / cam_h

    cam_uint8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(cam_uint8, (5, 5), 0)
    otsu_value, _ = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    effective_threshold = max(int(round(threshold * 255)), int(otsu_value))
    _, binary = cv2.threshold(cam_uint8, effective_threshold, 255, cv2.THRESH_BINARY)

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _MORPH_KERNEL_3x3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _MORPH_KERNEL_3x3, iterations=1)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if float(distance.max()) > 0.0:
        _, sure_fg = cv2.threshold(
            distance,
            _LESION_SPLIT_RATIO * float(distance.max()),
            255,
            0,
        )
        sure_fg = sure_fg.astype(np.uint8)
    else:
        sure_fg = binary.copy()

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    unknown = cv2.subtract(binary, sure_fg)
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(cam_uint8, cv2.COLOR_GRAY2BGR), markers)

    global_mean = float(cam.mean())
    denom = max(global_mean, 1e-6)

    regions: list[dict[str, Any]] = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        component_mask = (markers == label).astype(np.uint8)
        if component_mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        cx, cy, bw, bh = cv2.boundingRect(cnt)
        ox = int(round(cx * scale_x))
        oy = int(round(cy * scale_y))
        obw = int(round(bw * scale_x))
        obh = int(round(bh * scale_y))
        area = obw * obh
        if area < min_area:
            continue
        # Clip slice indices to stay within CAM bounds before computing mean.
        y_end = min(cy + bh, cam_h)
        x_end = min(cx + bw, cam_w)
        component_roi = component_mask[cy:y_end, cx:x_end].astype(bool)
        roi_activations = cam[cy:y_end, cx:x_end]
        if component_roi.any():
            mean_act = float(roi_activations[component_roi].mean())
        else:
            mean_act = float(roi_activations.mean())
        contour_area = float(cv2.contourArea(cnt))
        perimeter = float(cv2.arcLength(cnt, True))
        circularity = (
            (4.0 * np.pi * contour_area) / (perimeter * perimeter)
            if perimeter > 0.0
            else 0.0
        )
        circularity = float(np.clip(circularity, 0.0, 1.0))
        relative_intensity = mean_act / denom
        regions.append(
            {
                "x": ox,
                "y": oy,
                "width": obw,
                "height": obh,
                "mean_activation": round(mean_act, 4),
                "relative_intensity": round(relative_intensity, 4),
                "circularity": round(circularity, 4),
                "area": area,
            }
        )

    regions.sort(key=lambda r: r["mean_activation"], reverse=True)
    return regions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_gradcam(
    image: np.ndarray,
    source_path: str,
    model_path: str | None = None,
    target_class: int | None = None,
    overlay_alpha: float = _OVERLAY_ALPHA,
) -> dict[str, Any]:
    """Run Grad-CAM and Guided Grad-CAM on a processed fundus image.

    Generates two colour-coded heatmap overlays saved alongside the source
    image:

    * ``<stem>_gradcam.jpg`` – standard Grad-CAM overlay (coarse, class-
      discriminative blobs).
    * ``<stem>_guided_gradcam.jpg`` – Guided Grad-CAM overlay (pixel-precise
      edges, highlighting the exact boundaries of small retinal lesions such
      as microaneurysms and exudates).

    A JSON audit record is written to ``<stem>_gradcam.json`` containing the
    predicted DR grade, confidence score, paths to both overlays, and lesion
    regions with bounding-box coordinates plus morphology/intensity metrics.

    If *model_path* is not provided the value of the ``OPEN_DR_MODEL_PATH``
    environment variable is used.  When neither is set the model runs in
    demo mode with random weights.

    Parameters
    ----------
    image:
        Processed BGR fundus image (output of the extraction pipeline).
    source_path:
        Filesystem path of the processed image – used to derive output
        file names and recorded in the audit document.  **The caller is
        responsible for ensuring this path was generated by the openDR
        pipeline and has not been influenced by external input without
        prior sanitisation.**
    model_path:
        Optional path to a ``.pt`` / ``.pth`` checkpoint.  Falls back to
        the ``OPEN_DR_MODEL_PATH`` environment variable, then demo mode.
    target_class:
        Force Grad-CAM to explain a specific DR class index.  Defaults to
        the model's top-1 prediction.
    overlay_alpha:
        Heatmap opacity (0–1) used when blending both overlays.

    Returns
    -------
    dict
        The audit record that was written to ``<stem>_gradcam.json``.

    Raises
    ------
    RuntimeError
        If PyTorch or torchvision is not installed.
    ValueError
        If *source_path* does not point to an existing JPEG/PNG file, or
        if the derived output paths would escape the source directory.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch and torchvision are required for Grad-CAM. "
            "Install them with: pip install torch torchvision"
        )

    # Validate source path: must resolve to an existing file with an
    # image extension, and its parent directory must be a real directory.
    p = Path(source_path).resolve()
    if not p.is_file():
        raise ValueError("source_path does not point to an existing image file.")
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(
            f"Unexpected file extension {p.suffix!r} for source_path; "
            "expected .jpg, .jpeg, or .png."
        )

    # Derive output paths – all are forced into the same directory as the
    # source so they cannot escape to a different filesystem location.
    parent = p.parent
    overlay_path = str(parent / (p.stem + "_gradcam.jpg"))
    guided_overlay_path = str(parent / (p.stem + "_guided_gradcam.jpg"))
    json_path = str(parent / (p.stem + "_gradcam.json"))

    resolved_model_path = model_path or os.environ.get("OPEN_DR_MODEL_PATH")
    device = torch.device("cpu")
    model = _load_model(resolved_model_path, device=device)
    tensor = _preprocess(image).to(device)

    # Step 1 – standard Grad-CAM (coarse class-discriminative map).
    cam, pred_class, confidence = _compute_gradcam(model, tensor, target_class)

    # Step 2 – guided backpropagation (pixel-level gradients).
    guided_grads = _compute_guided_backprop(model, tensor, pred_class)

    # Step 3 – Guided Grad-CAM (pixel-precise, class-discriminative).
    guided_gradcam = _compute_guided_gradcam(cam, guided_grads)

    overlay = _overlay_heatmap(image, cam, alpha=overlay_alpha)
    guided_overlay = _overlay_heatmap(image, guided_gradcam, alpha=overlay_alpha)
    lesions = _extract_lesion_regions(cam, image.shape[:2])

    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(guided_overlay_path, guided_overlay)

    audit_record: dict[str, Any] = {
        "schema_version": "1.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_image": source_path,
        "gradcam_overlay": overlay_path,
        "guided_gradcam_overlay": guided_overlay_path,
        "gradcam_audit_json": json_path,
        "model_path": resolved_model_path or "random_weights_demo",
        "predicted_dr_grade": {
            "class_index": pred_class,
            "label": DR_CLASSES[pred_class],
            "confidence": round(confidence, 4),
        },
        "lesion_regions": lesions,
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(audit_record, fh, indent=2, ensure_ascii=False)

    return audit_record
