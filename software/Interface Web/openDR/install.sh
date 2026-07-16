#!/usr/bin/env bash
set -euo pipefail

echo "Installing openDR dependencies for Raspberry Pi OS (Bookworm/Bullseye)..."

sudo apt-get update
sudo apt-get -y upgrade

sudo apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  python3-opencv \
  python3-picamera2 \
  python3-pigpio \
  python3-flask \
  python3-numpy \
  python3-requests \
  pigpio \
  libcamera-apps \
  chromium-browser \
  nodejs \
  npm

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade imutils

# ── Tailwind CSS (compiled offline; output committed to static/css/tailwind.css)
echo "Building Tailwind CSS..."
cd /home/pi/openDR || { echo "ERROR: /home/pi/openDR not found. Ensure the repository is cloned there."; exit 1; }
npm install --save-dev tailwindcss
npx tailwindcss -i ./static/css/tailwind.src.css -o ./static/css/tailwind.css --minify
echo "Tailwind CSS built successfully."

echo "Done. Ensure libcamera is enabled and run: python3 /home/pi/openDR/fundus.py"
