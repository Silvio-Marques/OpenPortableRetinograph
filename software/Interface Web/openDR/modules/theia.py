"""
Client for the Theia diabetic-retinopathy grading API.

Uploads a processed fundus image to the MIT Media Lab Theia endpoint and
returns the numeric DR grade contained in the JSON response.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests


def grade_request(filename: str) -> float:
    """Upload a processed fundus image to Theia and return the DR grade.

    Reads an API key from ``<OPEN_DR_BASE>/key`` (the default base
    directory is ``/home/pi/openDR``, overridable via the
    ``OPEN_DR_BASE`` environment variable) and POSTs the image to the
    Theia REST endpoint.

    Parameters
    ----------
    filename:
        Path to the processed JPEG image to upload.

    Returns
    -------
    float
        The numeric diabetic-retinopathy grade from the API response, or
        ``-1`` if the key file is missing, the HTTP request fails, or the
        response cannot be parsed.
    """
    base_folder = Path(os.environ.get("OPEN_DR_BASE", "/home/pi/openDR")).resolve()
    key_path = base_folder / "key"

    try:
        with key_path.open("r", encoding="utf-8") as keyfile:
            key = keyfile.readline().strip()
    except IOError:
        print("CANNOT FIND KEY FOR THEIA. PLEASE CHECK.")
        return -1

    uri = "https://theia.media.mit.edu/api/v1/uploadImage?key=" + key
    with open(filename, "rb") as image_file:
        response = requests.post(uri, files={"file": image_file})

    if response.status_code == 200:
        data = json.loads(response.text)
        grade_value = data.get("grade")
        if isinstance(grade_value, list) and grade_value:
            return float(grade_value[0])
        return float(grade_value)
    return -1


## BEGIN THE REQUEST:
# print grade_request(open('normal1.jpg', 'rb'))

