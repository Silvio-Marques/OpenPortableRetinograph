##############################################################################
##  OWL v3.0                                                          ########
## ------------------------------------------------------------       ########
##  Authors: Ayush Yadav, Devesh Jain, Ebin Philip, Dhruv Joshi       ########
##  Revision: Helio Bentzen                                           ########
##############################################################################

import atexit
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import safe_join

from RetinaCamera import (
    CameraDisconnectedError,
    CameraOverheatError,
    RetinaCamera,
    RetinaCameraError,
)
from modules.process import (
    DEFAULT_PROCESSING_SETTINGS,
    grade,
    grade_with_explanation,
    normalize_processing_settings,
)

try:
    import pigpio
except ImportError:  # pragma: no cover - depends on Raspberry Pi runtime
    pigpio = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
BASE_FOLDER = Path(os.environ.get("OPEN_DR_BASE", "/home/pi/openDR")).resolve()
TOKENS = ["Flip", "Vid", "Click", "Switch", "Grade", "Explain", "Shut"]
PATIENT_ID_RE = re.compile(r"^[A-Z0-9_-]{1,64}$")
FOCUS_WARNING_MESSAGE = "Posicione o paciente e foque antes de capturar"
CAMERA_UNAVAILABLE_MESSAGE = (
    "Câmera indisponível - você ainda pode enviar fotos para análise"
)
MIN_FOCUS_SCORE = 140
DARK_PIXEL_THRESHOLD = 58
MIN_DARK_PIXELS = 120
MIN_DARK_DENSITY = 0.20
MIN_DARK_RATIO = 0.01
MAX_DARK_RATIO = 0.42
PREVIEW_MIN_INTERVAL_S = 0.20
INFERENCE_WORKER_COUNT = max(1, int(os.environ.get("OPEN_DR_INFERENCE_WORKERS", "2")))
MAX_INFERENCE_JOB_HISTORY = max(
    1, int(os.environ.get("OPEN_DR_MAX_INFERENCE_JOB_HISTORY", "8"))
)
INFERENCE_STEPS = (
    ("received", "Imagem recebida"),
    ("preprocessing", "Pré-processamento (limpeza de ruído)"),
    ("inference", "Inferência do modelo"),
    ("report", "Geração do relatório"),
)
CAPTURE_FILENAME_RE = re.compile(
    r"^(?P<patient_id>.+)_(?P<date>\d{8})_(?P<time>\d{12})_(?P<uuid>[0-9a-f]{32})_(?P<capture>\d+)\.jpg$"
)
GALLERY_DEFAULT_PAGE_SIZE = 12
GALLERY_MAX_PAGE_SIZE = 60
THUMBNAIL_DEFAULT_SIZE = 256
THUMBNAIL_MIN_SIZE = 96
THUMBNAIL_MAX_SIZE = 384
UPLOAD_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

orangeyellow = 14
bluegreen = 15
switch = 4
pi = None


def default_processing_settings():
    return dict(DEFAULT_PROCESSING_SETTINGS)


class CameraSessionState:
    def __init__(self):
        self.lock = Lock()
        self.camera = None
        self.camera_error = None
        self.last_img = None
        self.inference_job_id = None
        self.patient_id = ""
        self.processing_settings = default_processing_settings()

    def reset(self):
        self.inference_job_id = None
        self.patient_id = ""
        self.last_img = None
        self.camera_error = None
        self.processing_settings = default_processing_settings()

    def stop_camera(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera = None

    def deactivate_camera(self, error_message):
        """Stop and clear the active camera after a hardware fault.

        Caller must hold ``self.lock``. Keeps the patient session (and thus
        photo upload) alive even though camera-based capture is now
        unavailable.
        """
        self.stop_camera()
        self.camera_error = error_message


state = CameraSessionState()
preview_rate_lock = Lock()
preview_last_request = {}
inference_jobs_lock = Lock()
inference_jobs = {}
inference_executor = ThreadPoolExecutor(
    max_workers=INFERENCE_WORKER_COUNT,
    thread_name_prefix="inference",
)


def get_processing_settings():
    with state.lock:
        return dict(state.processing_settings)


def update_processing_settings_from_request(form_data):
    current_settings = get_processing_settings()
    updated_settings = normalize_processing_settings(
        {
            "brightness": form_data.get("brightness", current_settings["brightness"]),
            "contrast": form_data.get("contrast", current_settings["contrast"]),
            "fundus_threshold": form_data.get(
                "fundus_threshold", current_settings["fundus_threshold"]
            ),
            "glare_threshold": form_data.get(
                "glare_threshold", current_settings["glare_threshold"]
            ),
        }
    )
    with state.lock:
        state.processing_settings = updated_settings
    return updated_settings


@atexit.register
def shutdown_inference_executor():
    inference_executor.shutdown(wait=False, cancel_futures=True)


def describe_camera_error(exc):
    """Translate a RetinaCameraError into a user-facing Portuguese message."""
    if isinstance(exc, CameraOverheatError):
        return "CÂMERA SUPERAQUECIDA - AGUARDE E TENTE NOVAMENTE"
    if isinstance(exc, CameraDisconnectedError):
        return "CÂMERA DESCONECTADA - VERIFIQUE O CABO"
    return CAMERA_UNAVAILABLE_MESSAGE


def start_camera_safely():
    """Start the retinal camera, tolerating missing or faulty hardware.

    Returns a ``(camera, error_message)`` pair where exactly one of the two
    is truthy. This lets a patient session proceed for photo-upload-only use
    when no camera is attached (e.g. a workstation without the capture rig).
    """
    try:
        return RetinaCamera().start(), None
    except RetinaCameraError as exc:
        app.logger.warning(
            "Camera unavailable, continuing session without it: %s", exc
        )
        return None, describe_camera_error(exc)


@app.route("/")
def my_form():
    normalON()
    return render_template("index.html")


@app.route("/", methods=["POST"])
def my_form_post():
    patient_id = sanitize_patient_id(request.form["text"].upper())
    if not patient_id:
        return render_template("index.html")

    make_a_dir(patient_id)
    camera, camera_error = start_camera_safely()
    with state.lock:
        state.stop_camera()
        state.patient_id = patient_id
        state.inference_job_id = None
        state.last_img = None
        state.camera = camera
        state.camera_error = camera_error
    return redirect(url_for("captureSimpleFunc"))


@app.route("/captureSimple", methods=["GET", "POST"])
def captureSimpleFunc():
    if request.method == "GET":
        return render_capture()

    processing_settings = update_processing_settings_from_request(request.form)

    if "d" not in request.form:
        return render_capture()

    d = request.form["d"]

    if d == "Click":
        if request.form.get("focus_ok") != "1":
            return render_capture(FOCUS_WARNING_MESSAGE)
        with state.lock:
            if state.camera is None or not state.patient_id:
                return render_capture("NO ACTIVE CAPTURE SESSION")
            try:
                image = state.camera.capture()
            except RetinaCameraError as exc:
                error_message = describe_camera_error(exc)
                state.deactivate_camera(error_message)
                return render_capture(error_message)
            if not is_eye_in_focus(image):
                return render_capture(FOCUS_WARNING_MESSAGE)
            state.last_img = save_captured_images(state.patient_id, image)
            state.inference_job_id = None
        return render_capture()

    if d == "Flip":
        with state.lock:
            if state.camera is None:
                return render_capture("NO ACTIVE CAPTURE SESSION")
            state.camera.flip_cam()
        return render_capture()

    if d == "Vid":
        with state.lock:
            if state.camera is None or not state.patient_id:
                return render_capture("NO ACTIVE CAPTURE SESSION")
            state.camera.continuous_capture()
            if not state.camera.wait_for_capture(timeout=5):
                return render_capture("CAPTURE TIMEOUT - RETRY OR CHECK CAMERA")
            if not state.camera.images:
                # RetinaCamera swallows hardware errors inside its background
                # thread and just stops early, so an empty result here means
                # the camera failed mid-recording rather than timed out.
                error_message = "FALHA NA CÂMERA DURANTE A GRAVAÇÃO"
                state.deactivate_camera(error_message)
                return render_capture(error_message)
            state.last_img = save_captured_images(state.patient_id, state.camera.images)
            state.inference_job_id = None
        return render_capture()

    if d == "Grade":
        with state.lock:
            last_img = state.last_img
        if last_img is None:
            return render_capture("NO IMAGE SPECIFIED")

        grade_result = str(grade(last_img, processing_settings=processing_settings))[:4]
        print("the grade is " + grade_result)
        return render_capture(grade_result)

    if d == "Explain":
        with state.lock:
            last_img = state.last_img
            active_job_id = state.inference_job_id
        if last_img is None:
            return render_capture("NO IMAGE SPECIFIED")

        if active_job_id and is_inference_job_running(active_job_id):
            return render_capture("INFERENCE IN PROGRESS", inference_job_id=active_job_id)

        job_id = create_inference_job(last_img)
        with state.lock:
            state.inference_job_id = job_id

        inference_executor.submit(
            run_explanation_job,
            job_id,
            last_img,
            dict(processing_settings),
        )
        return render_capture("PROCESSANDO...", inference_job_id=job_id)

    if d == "Switch":
        with state.lock:
            had_session = bool(state.patient_id)
            state.stop_camera()
            state.reset()
        if had_session:
            return redirect(url_for("my_form"))
        return render_capture()

    if d == "Shut":
        shut_down()
        return render_capture()

    return render_capture()


@app.route("/inference-status/<job_id>", methods=["GET"])
def inference_status(job_id):
    job = get_inference_job(job_id)
    if job is None:
        return jsonify({"error": "INFERENCE JOB NOT FOUND"}), 404

    return jsonify(serialize_inference_job(job))


@app.route("/preview-frame", methods=["GET"])
def preview_frame():
    client_key = request.remote_addr or "unknown"
    now = time.monotonic()
    with preview_rate_lock:
        last_request = preview_last_request.get(client_key, 0.0)
        if now - last_request < PREVIEW_MIN_INTERVAL_S:
            abort(429)
        preview_last_request[client_key] = now

    with state.lock:
        if state.camera is None:
            abort(404)
        try:
            preview = state.camera.capture_preview()
        except RetinaCameraError as exc:
            state.deactivate_camera(describe_camera_error(exc))
            abort(503)
    response = Response(preview.tobytes(), mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/upload-image", methods=["POST"])
def upload_image():
    """Accept a user-uploaded fundus photo and make it the active image.

    The uploaded file is validated (extension, size, decodability), decoded
    with OpenCV to normalise any supported format to a JPEG on disk, and saved
    into the current patient's capture set via :func:`save_captured_images`.
    It then becomes ``state.last_img`` so it can be graded with the existing
    "Grade" / "Explain" controls, exactly like a camera capture.
    """
    with state.lock:
        patient_id = state.patient_id
    if not patient_id:
        return render_capture("NENHUMA SESSÃO ATIVA - INICIE UMA SESSÃO")

    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return render_capture("NENHUM ARQUIVO ENVIADO")

    if Path(uploaded.filename).suffix.lower() not in UPLOAD_ALLOWED_EXTENSIONS:
        return render_capture("FORMATO INVÁLIDO - ENVIE JPG OU PNG")

    file_bytes = uploaded.read()
    if not file_bytes:
        return render_capture("ARQUIVO VAZIO")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        return render_capture("ARQUIVO MUITO GRANDE (MÁX. 25 MB)")

    image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return render_capture("IMAGEM INVÁLIDA OU CORROMPIDA")

    with state.lock:
        if not state.patient_id:
            return render_capture("NENHUMA SESSÃO ATIVA - INICIE UMA SESSÃO")
        state.last_img = save_captured_images(state.patient_id, image)
        state.inference_job_id = None
    return render_capture("IMAGEM ENVIADA")


def is_eye_in_focus(image_buffer):
    gray_frame = cv2.imdecode(image_buffer, cv2.IMREAD_GRAYSCALE)
    if gray_frame is None:
        return False

    resized = cv2.resize(gray_frame, (176, 132), interpolation=cv2.INTER_AREA)
    focus_score = cv2.Laplacian(resized, cv2.CV_64F).var()

    _, dark_regions = cv2.threshold(
        resized,
        DARK_PIXEL_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV,
    )
    dark_pixel_count = int(cv2.countNonZero(dark_regions))
    if dark_pixel_count < MIN_DARK_PIXELS:
        return False

    contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    x, y, width, height = cv2.boundingRect(largest_contour)
    bounding_area = max(1, width * height)
    dark_density = contour_area / bounding_area
    dark_ratio = dark_pixel_count / float(resized.shape[0] * resized.shape[1])
    has_eye_contour = (
        dark_density > MIN_DARK_DENSITY
        and dark_ratio > MIN_DARK_RATIO
        and dark_ratio < MAX_DARK_RATIO
    )
    return focus_score >= MIN_FOCUS_SCORE and has_eye_contour


def render_capture(
    grade_message="",
    overlay_filename=None,
    json_filename=None,
    dr_label=None,
    confidence=None,
    lesion_count=None,
    inference_job_id=None,
):
    with state.lock:
        patient_id = state.patient_id
        camera_available = state.camera is not None
        camera_warning = state.camera_error
    return render_template(
        "capture_simple.html",
        params=TOKENS,
        grades={"grade": grade_message},
        focus_warning_message=FOCUS_WARNING_MESSAGE,
        overlay_filename=overlay_filename,
        json_filename=json_filename,
        dr_label=dr_label,
        confidence=confidence,
        lesion_count=lesion_count,
        inference_job_id=inference_job_id,
        patient_id=patient_id,
        camera_available=camera_available,
        camera_warning=camera_warning,
        gallery_page_size=GALLERY_DEFAULT_PAGE_SIZE,
        processing_settings=get_processing_settings(),
        processing_defaults=default_processing_settings(),
        max_upload_bytes=MAX_UPLOAD_BYTES,
        upload_allowed_extensions=sorted(UPLOAD_ALLOWED_EXTENSIONS),
    )


def format_grade_display(grade_value):
    if grade_value is None:
        return ""
    return str(grade_value)[:4]


def create_inference_job(image_path):
    raw_filename = Path(image_path).name
    job_id = str(uuid4())
    steps = {
        key: {
            "key": key,
            "label": label,
            "status": "pending",
            "detail": "",
            "image_filename": raw_filename if key == "received" else None,
        }
        for key, label in INFERENCE_STEPS
    }
    steps["received"]["status"] = "completed"
    steps["preprocessing"]["status"] = "running"

    with inference_jobs_lock:
        inference_jobs[job_id] = {
            "job_id": job_id,
            "created_at": time.time(),
            "status": "running",
            "current_step": "preprocessing",
            "message": "Imagem enviada para a fila de inferência.",
            "error": "",
            "grade_message": "",
            "steps": steps,
            "result": {},
        }
        prune_inference_jobs_locked()

    return job_id


def prune_inference_jobs_locked():
    """Prune the oldest completed or failed jobs while holding the job lock."""
    terminal_job_ids = [
        job_id
        for job_id, job in sorted(
            inference_jobs.items(), key=lambda item: item[1].get("created_at", 0.0)
        )
        if job["status"] != "running"
    ]
    prune_count = len(terminal_job_ids) - MAX_INFERENCE_JOB_HISTORY
    if prune_count <= 0:
        return
    for job_id in terminal_job_ids[:prune_count]:
        inference_jobs.pop(job_id, None)


def is_inference_job_running(job_id):
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        return job is not None and job["status"] == "running"


def get_inference_job(job_id):
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        return deepcopy(job) if job is not None else None


def advance_inference_job(job_id, completed_step, next_step=None, **payload):
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is None:
            return

        completed_state = job["steps"][completed_step]
        completed_state["status"] = "completed"

        processed_path = payload.get("processed_path")
        if completed_step == "preprocessing" and processed_path:
            completed_state["image_filename"] = Path(processed_path).name
            completed_state["detail"] = "Ruído removido e imagem preparada."
            job["message"] = "Pré-processamento concluído."
        elif completed_step == "inference":
            job["grade_message"] = format_grade_display(payload.get("theia_grade"))
            completed_state["detail"] = (
                f"Resultado do modelo: {job['grade_message'] or 'N/A'}"
            )
            job["message"] = "Inferência do modelo concluída."
        elif completed_step == "report":
            gradcam_record = payload["gradcam"]
            overlay_filename = Path(gradcam_record["gradcam_overlay"]).name
            json_filename = Path(gradcam_record["gradcam_audit_json"]).name
            job["grade_message"] = format_grade_display(payload.get("theia_grade"))
            completed_state["image_filename"] = overlay_filename
            completed_state["detail"] = "Relatório final disponível."
            job["result"] = {
                "overlay_filename": overlay_filename,
                "json_filename": json_filename,
                "dr_label": gradcam_record["predicted_dr_grade"]["label"],
                "confidence": gradcam_record["predicted_dr_grade"]["confidence"],
                "lesion_count": len(gradcam_record["lesion_regions"]),
            }
            job["message"] = "Relatório final gerado."

        if next_step is not None:
            job["current_step"] = next_step
            job["steps"][next_step]["status"] = "running"
        else:
            job["current_step"] = None
            job["status"] = "completed"


def fail_inference_job(job_id, error_message):
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is None:
            return

        current_step = job.get("current_step")
        if current_step:
            job["steps"][current_step]["status"] = "failed"
        job["status"] = "failed"
        job["error"] = error_message
        job["message"] = error_message
        job["current_step"] = None


def serialize_inference_job(job):
    result = job["result"]
    serialized_steps = []
    for step_key, step_label in INFERENCE_STEPS:
        step = job["steps"][step_key]
        image_filename = step.get("image_filename")
        serialized_steps.append(
            {
                "key": step_key,
                "label": step_label,
                "status": step["status"],
                "detail": step.get("detail", ""),
                "image_url": (
                    url_for("serve_image", filename=image_filename)
                    if image_filename
                    else None
                ),
            }
        )

    overlay_filename = result.get("overlay_filename")
    json_filename = result.get("json_filename")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "current_step": job["current_step"],
        "message": job["message"],
        "error": job["error"],
        "grade_message": job["grade_message"],
        "steps": serialized_steps,
        "result": {
            "overlay_image_url": (
                url_for("serve_image", filename=overlay_filename)
                if overlay_filename
                else None
            ),
            "json_url": (
                url_for("serve_image", filename=json_filename)
                if json_filename
                else None
            ),
            "dr_label": result.get("dr_label"),
            "confidence": result.get("confidence"),
            "lesion_count": result.get("lesion_count"),
        },
    }


def run_explanation_job(job_id, image_path, processing_settings):
    def status_callback(step_name, **payload):
        if step_name == "preprocessing":
            advance_inference_job(job_id, "preprocessing", next_step="inference", **payload)
        elif step_name == "inference":
            advance_inference_job(job_id, "inference", next_step="report", **payload)
        elif step_name == "report":
            advance_inference_job(job_id, "report", **payload)

    try:
        grade_with_explanation(
            image_path,
            status_callback=status_callback,
            processing_settings=processing_settings,
        )
    except RuntimeError as exc:
        app.logger.exception("Inference job %s failed during processing.", job_id)
        fail_inference_job(job_id, f"GRAD-CAM ERROR: {exc}")
    except OSError as exc:  # pragma: no cover - runtime safeguards
        app.logger.exception("Inference job %s failed with OSError.", job_id)
        fail_inference_job(job_id, f"FILE ERROR: {exc}")
    except ValueError as exc:  # pragma: no cover - runtime safeguards
        app.logger.exception("Inference job %s failed with ValueError.", job_id)
        fail_inference_job(job_id, f"DATA ERROR: {exc}")
    except KeyError as exc:  # pragma: no cover - runtime safeguards
        app.logger.exception("Inference job %s failed with KeyError.", job_id)
        fail_inference_job(job_id, f"REPORT ERROR: missing field {exc}")


@app.route("/images/<path:filename>")
def serve_image(filename):
    """Serve captured and processed images from the patient images directory.

    The filename is validated to reject path-traversal attempts before
    handing it to :func:`~flask.send_from_directory`.
    """
    # Extract only the bare filename component to prevent traversal outside
    # the images directory (e.g. "../../etc/passwd" → rejected).
    try:
        image_path = resolved_media_path(filename)
    except ValueError:
        app.logger.warning("Rejected path-traversal attempt in /images: %s", filename)
        abort(400)
    return send_from_directory(str(images_directory()), image_path.name)


@app.route("/thumbnails/<path:filename>")
def serve_thumbnail(filename):
    try:
        source_path = resolved_media_path(filename, allowed_suffixes={".jpg", ".jpeg"})
    except ValueError:
        app.logger.warning(
            "Rejected path-traversal attempt in /thumbnails: %s", filename
        )
        abort(400)

    if not source_path.exists() or not source_path.is_file():
        abort(404)

    requested_size = request.args.get("w", THUMBNAIL_DEFAULT_SIZE, type=int)
    size = max(THUMBNAIL_MIN_SIZE, min(THUMBNAIL_MAX_SIZE, requested_size))
    thumbnail_bytes = cached_thumbnail_bytes(
        source_path.name,
        size,
        source_path.stat().st_mtime_ns,
    )
    if thumbnail_bytes is None:
        return send_from_directory(str(images_directory()), source_path.name)

    response = Response(thumbnail_bytes, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "public, max-age=300"
    return response


@app.route("/capture-gallery", methods=["GET"])
def capture_gallery():
    with state.lock:
        patient_id = state.patient_id

    if not patient_id:
        return jsonify(
            {
                "patient_id": "",
                "page": 1,
                "page_size": GALLERY_DEFAULT_PAGE_SIZE,
                "total": 0,
                "has_more": False,
                "items": [],
            }
        )

    page = request.args.get("page", default=1, type=int) or 1
    page = max(1, page)
    page_size = request.args.get(
        "page_size",
        default=GALLERY_DEFAULT_PAGE_SIZE,
        type=int,
    ) or GALLERY_DEFAULT_PAGE_SIZE
    page_size = max(1, min(GALLERY_MAX_PAGE_SIZE, page_size))

    all_items = list_patient_capture_metadata(patient_id)
    total = len(all_items)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = all_items[start:end]

    return jsonify(
        {
            "patient_id": patient_id,
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_more": end < total,
            "items": page_items,
        }
    )


def save_captured_images(patient_id, images):
    no = 1
    patient_id = validated_patient_id(patient_id)
    last_saved_path = None

    if isinstance(images, list):
        for img in images:
            image_path = write_captured_image(patient_id, no, img)
            last_saved_path = str(image_path)
            no += 1
    else:
        image_path = write_captured_image(patient_id, no, images)
        last_saved_path = str(image_path)

    return last_saved_path


def write_captured_image(patient_id, capture_number, image_buffer):
    """Persist one captured image with a direct JPEG write when possible.

    Picamera2 captures already arrive as JPEG byte buffers for the current
    Flask flow, so those buffers are written directly to disk to avoid an
    unnecessary decode/re-encode round trip on the Raspberry Pi CPU. If a
    decoded image matrix is provided, OpenCV falls back to encoding it.
    """
    image_filename = build_image_filename(patient_id, capture_number)
    image_path = images_directory() / image_filename

    if isinstance(image_buffer, (bytes, bytearray)):
        with open_captured_image_file(image_filename) as image_file:
            image_file.write(image_buffer)
        return image_path

    # Picamera2 / OpenCV encoded JPEG buffers are 1-D byte arrays.
    if hasattr(image_buffer, "ndim") and image_buffer.ndim == 1 and hasattr(
        image_buffer, "tobytes"
    ):
        with open_captured_image_file(image_filename) as image_file:
            image_file.write(image_buffer.tobytes())
        return image_path

    wrote_image = cv2.imwrite(str(image_path), image_buffer)
    if not wrote_image:
        raise OSError(build_image_write_error(image_path, image_buffer))
    return image_path


def build_image_write_error(image_path, image_buffer):
    buffer_shape = getattr(image_buffer, "shape", None)
    return (
        "Unable to write captured image to "
        f"{image_path} (type={type(image_buffer).__name__}, shape={buffer_shape})."
    )


def images_directory():
    return (BASE_FOLDER / "images").resolve()


def validated_media_filename(value, allowed_suffixes=None):
    safe_name = Path(value).name
    if safe_name != value or safe_name in {"", ".", ".."}:
        raise ValueError(f"Invalid media filename: {value}")
    if allowed_suffixes and Path(safe_name).suffix.lower() not in allowed_suffixes:
        raise ValueError(f"Invalid media filename: {value}")
    return safe_name


def resolved_media_path(value, allowed_suffixes=None):
    safe_name = validated_media_filename(value, allowed_suffixes=allowed_suffixes)
    # safe_join returns None when a path would escape the base directory.
    joined_path = safe_join(str(images_directory()), safe_name)
    if joined_path is None:
        raise ValueError(f"Invalid media filename: {value}")
    return Path(joined_path)


def parse_capture_filename(filename):
    match = CAPTURE_FILENAME_RE.fullmatch(filename)
    if match is None:
        return None

    capture_time = f"{match.group('date')}_{match.group('time')}"
    try:
        captured_at = datetime.strptime(capture_time, "%Y%m%d_%H%M%S%f").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None

    return {
        "patient_id": match.group("patient_id"),
        "capture_number": int(match.group("capture")),
        "captured_at": captured_at,
    }


def read_capture_result(image_path):
    report_path = image_path.with_name(f"{image_path.stem}_processed_gradcam.json")
    if not report_path.exists():
        return None
    try:
        with report_path.open("r", encoding="utf-8") as report_file:
            payload = json.load(report_file)
    except (OSError, json.JSONDecodeError):
        return None

    predicted_grade = payload.get("predicted_dr_grade") or {}
    lesions = payload.get("lesion_regions")
    lesion_count = len(lesions) if isinstance(lesions, list) else None
    confidence = predicted_grade.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = None

    return {
        "dr_label": predicted_grade.get("label"),
        "confidence": confidence,
        "lesion_count": lesion_count,
        "json_url": url_for("serve_image", filename=report_path.name),
    }


def list_patient_capture_metadata(patient_id):
    patient_id = validated_patient_id(patient_id)
    directory = images_directory()
    if not directory.exists():
        return []

    metadata = []
    for image_path in directory.glob(f"{patient_id}_*.jpg"):
        parsed = parse_capture_filename(image_path.name)
        if parsed is None or parsed["patient_id"] != patient_id:
            continue

        result = read_capture_result(image_path)
        metadata.append(
            {
                "filename": image_path.name,
                "patient_id": patient_id,
                "capture_number": parsed["capture_number"],
                "captured_at": parsed["captured_at"].isoformat(),
                "image_url": url_for("serve_image", filename=image_path.name),
                "thumbnail_url": url_for("serve_thumbnail", filename=image_path.name),
                "result": {
                    "status": "ready" if result else "pending",
                    "dr_label": result.get("dr_label") if result else None,
                    "confidence": result.get("confidence") if result else None,
                    "lesion_count": result.get("lesion_count") if result else None,
                    "json_url": result.get("json_url") if result else None,
                },
            }
        )

    metadata.sort(key=lambda item: item["captured_at"], reverse=True)
    return metadata


@lru_cache(maxsize=256)
def cached_thumbnail_bytes(filename, max_dimension, modified_ns):
    # Keep modified_ns in the cache key so updated files invalidate cached bytes.
    _ = modified_ns
    image_path = images_directory() / filename
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        return None

    height, width = frame.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension > max_dimension:
        scale = max_dimension / float(largest_dimension)
        resized = cv2.resize(
            frame,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = frame

    encoded_ok, encoded = cv2.imencode(
        ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 72]
    )
    if not encoded_ok:
        return None
    return encoded.tobytes()


def open_captured_image_file(image_filename):
    safe_name = Path(image_filename).name
    if safe_name != image_filename or safe_name in {"", ".", ".."}:
        raise ValueError(f"Invalid image filename: {image_filename}")
    output_path = (images_directory() / safe_name).resolve()
    try:
        output_path.relative_to(images_directory())
    except ValueError as exc:
        raise ValueError(f"Invalid image filename: {image_filename}") from exc
    return output_path.open("wb")


def build_image_filename(patient_id, capture_number):
    patient_id = validated_patient_id(patient_id)
    image_identifier = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S%f')}_{uuid4().hex}"
    )
    return f"{patient_id}_{image_identifier}_{capture_number}.jpg"


def make_a_dir(pr_t):
    validated_patient_id(pr_t)
    directory = BASE_FOLDER / "images"
    directory.mkdir(parents=True, exist_ok=True)


def sanitize_patient_id(value):
    cleaned = re.sub(r"[^A-Z0-9_-]", "", value)
    if PATIENT_ID_RE.fullmatch(cleaned):
        return cleaned
    return ""


def validated_patient_id(value):
    if not PATIENT_ID_RE.fullmatch(value):
        raise ValueError("Invalid patient identifier.")
    return value


def init_gpio():
    global pi

    if pigpio is None:
        return None

    if pi is None:
        controller = pigpio.pi()
        if not controller.connected:
            app.logger.warning("pigpio daemon is unavailable; GPIO output is disabled.")
            try:
                controller.stop()
            except Exception:  # pragma: no cover - best-effort cleanup on hardware init
                pass
            return None
        controller.set_mode(orangeyellow, pigpio.OUTPUT)
        controller.set_mode(bluegreen, pigpio.OUTPUT)
        controller.set_mode(switch, pigpio.INPUT)
        controller.set_pull_up_down(switch, pigpio.PUD_UP)
        pi = controller

    return pi


def normalON():
    controller = init_gpio()
    if controller is None:
        return
    controller.write(orangeyellow, 0)
    controller.write(bluegreen, 1)


def secondaryON():
    controller = init_gpio()
    if controller is None:
        return
    controller.write(orangeyellow, 1)
    controller.write(bluegreen, 0)


def shut_down():
    command = "/usr/bin/sudo /sbin/shutdown now"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]
    print(output.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    init_gpio()
    app.run(host="0.0.0.0", threaded=True)
