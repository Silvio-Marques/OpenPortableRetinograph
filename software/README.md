# Software

The **Open Portable Retinograph (OPR)** software is responsible for hardware control, retinal image acquisition, user interaction, and the execution of image processing and Artificial Intelligence algorithms.

The software architecture was designed with a modular approach, allowing each component to be developed and maintained independently.

---

# Directory Structure

```text
software/
├── Web Interface/
│   └── openDR/
│
├── RaspberryPi/
│
├── glaucoma/
│
└── README.md
```

---

# Directories

## Web Interface/openDR/

Contains the graphical user interface used to operate the retinograph.

**Key Features**

* Real-time camera preview
* Image acquisition
* Browser-based remote control
* Wi-Fi communication with the Raspberry Pi
* Image storage
* Responsive interface for smartphones and desktop computers
* Integration with the image processing pipeline

**Technologies**

* Python
* Flask
* HTML
* CSS
* JavaScript
* OpenCV

---

## RaspberryPi/

Contains all software executed directly on the Raspberry Pi.

**Responsibilities**

* System initialization
* M12 camera control
* LED illumination control
* GPIO communication
* Storage management
* Wi-Fi network configuration
* Communication with the Web Interface

---

## glaucoma/

Contains the Artificial Intelligence algorithms used for retinal image analysis.

This directory includes:

* Trained models
* Image preprocessing
* Segmentation
* Classification
* Grad-CAM visualization generation
* Local inference

The models can be executed locally on the Raspberry Pi or remotely through the Hugging Face Spaces platform.

---

# Software Workflow

```text
Raspberry Pi
      │
      ▼
Web Interface
      │
      ▼
Image Acquisition
      │
      ▼
Preprocessing
      │
      ▼
Artificial Intelligence
      │
      ▼
Results
```

---

# Dependencies

* Python 3.11+
* Flask
* OpenCV
* NumPy
* Picamera2
* libcamera
* pigpio
* Requests
* PyTorch
* Torchvision
* TensorFlow Lite

---

# Execution

1. Power on the Raspberry Pi.
2. Start the Flask server.
3. Connect a smartphone or computer via Wi-Fi.
4. Access the Web Interface.
5. Capture a retinal image.
6. Execute image processing and AI inference.

---

# Development

Each module can be updated independently.

The modular directory structure simplifies:

* Maintenance
* Testing
* Continuous integration
* Component reuse
* Collaborative development

---

# Related Projects

* OpenDR
* Hugging Face Spaces (Glaucoma)
* Raspberry Pi OS
* OpenCV
