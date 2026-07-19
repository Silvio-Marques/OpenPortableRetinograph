# Open Portable Retinograph (OPR)

**Development of an Open-Source Portable Raspberry Pi-Based Retinograph for Retinal Image Acquisition and Automated Retinopathy Screening**

---

# Authors

**Developed by**

Silvio Marques (IFPE)

**Advisors**

Tiago Lima (IFPE)

Helio Bentzen (IFPE)

David Ribeiro (IFPE)

**Institution**

IFPE – Palmares Campus

Maker Laboratory – Palmares Campus

Brazil

---

# About the Project

The **Open Portable Retinograph (OPR)** is an open-hardware project developed to provide a portable, low-cost platform for retinal fundus image acquisition.

The project combines additive manufacturing, commercially available optical components, and a Raspberry Pi-based embedded platform to enable retinal image acquisition using open-source hardware, with a focus on research, education, and future automated retinopathy screening applications.

The system architecture was designed to reduce the cost of conventional retinal imaging devices, facilitate replication by other institutions, and provide a platform for the future integration of embedded Artificial Intelligence algorithms.

---

# License

This project is distributed under the **MIT License**.

The CAD files, technical documentation, source code, and computational models may be used, modified, and redistributed under the terms of the license.

---

# Disclaimer

This equipment is intended exclusively for research, technological development, and educational purposes.

The prototype **is not certified for clinical use** and must not be used for medical diagnosis without the appropriate approval from the relevant regulatory authorities.

---

# Software Ecosystem

The retinograph was designed with a modular architecture, allowing integration with different Artificial Intelligence systems for the analysis of acquired retinal images.

## OpenDR

Automated Diabetic Retinopathy screening system.

**Repository:**

https://github.com/heliobentzen/openDR

**Key Features**

* Automated image acquisition
* Web interface
* TensorFlow Lite
* Grad-CAM
* Report generation
* Automatic classification

---

## Glaucoma AI

Deep Learning–based system for glaucoma classification from retinal images.

**Demo:**

https://huggingface.co/spaces/tiagopessoalim/glaucoma

**Intended Applications**

* Diagnostic second opinion
* Glaucoma screening support

---

# Version

**Current Version**

**v1.0.0 (Functional Prototype)**

---

# Version 1.0 Highlights

* Complete mechanical architecture developed
* Fully parametric CAD design
* Optical system based on a 20D condensing lens
* Integration of first-surface mirrors
* Structure manufactured using 3D printing
* TPU facial support
* Raspberry Pi 4 fully integrated into the system
* M12 camera integration
* High-power LED illumination system
* Web-based remote operation interface
* Wi-Fi communication between the Raspberry Pi and a smartphone
* Retinal image acquisition
* Functional prototype validated under laboratory bench conditions

---

# Objectives

* Develop an open-source portable retinograph
* Reduce the cost of retinal imaging equipment
* Facilitate platform replication
* Enable future embedded Artificial Intelligence applications
* Support research in Ophthalmology and Biomedical Engineering

---

# Features

## Mechanical Structure

* Complete CAD design
* 3D-printed structure
* Modular architecture
* Focus adjustment through optical housing displacement
* TPU facial support

---

## Optical System

* 20-diopter condensing lens
* First-surface mirrors
* M12 lens
* Anti-reflective internal coating
* Adjustable optical alignment

---

## Hardware

* Raspberry Pi 4
* M12 camera module
* High-power LEDs
* Rechargeable battery
* Wi-Fi connectivity

---

## User Interface

* Remote image acquisition
* Real-time visualization
* Image storage
* Browser-based control
* Compatible with smartphones and computers

---

# Prototype Specifications

| Feature              |           Value |
| -------------------- | --------------: |
| Development time     |        5 months |
| Weight               |          ~800 g |
| Cost                 |          US$491 |
| Structure            | PLA + TPU + MDF |
| Communication        |           Wi-Fi |
| Interface            |             Web |
| Battery life         |     ~40 minutes |
| Boot time            |      ~2 minutes |
| Simultaneous devices |               2 |

---

# Project Organization

```text
OpenPortableRetinograph/
│
├── hardware/
│   Files related to hardware development,
│   including CAD models, STL files, technical drawings,
│   bill of materials (BOM), schematics,
│   and prototype assembly documentation.
│
├── software/
│   Source code used by the retinograph,
│   including Raspberry Pi applications,
│   the web interface,
│   and image processing algorithms.
│
├── images/
│   Development photographs,
│   prototype images,
│   rendered CAD models,
│   and experimental results obtained
│   during equipment testing.
│
└── README.md
    Main project documentation.
```

As the project evolves, additional directories may be added for technical documentation, Artificial Intelligence models, scientific publications, and experimental results.

---

# Current Status

| Module                    | Status |
| ------------------------- | :----: |
| CAD Design                |    ✅   |
| Optical System            |    ✅   |
| 3D Printing               |    ✅   |
| Raspberry Pi Integration  |    ✅   |
| Web Interface             |    ✅   |
| Retinal Image Acquisition |    ✅   |
| Optical Testing           |    ✅   |
| Artificial Intelligence   |        |
| Clinical Validation       |        |

---

# Future Work

* Optical system optimization
* Development of a miniaturized version
* TensorFlow Lite model integration
* Automatic retinal segmentation
* Automated retinopathy classification
* Dedicated mobile application
* Clinical validation
* Open-hardware publication

---

# Acknowledgments

The authors would like to thank the institutions and collaborators involved in the development of this project, as well as the Open Hardware community and the Open Indirect Ophthalmoscope (OIO), Raspberry Pi Foundation, and OpenCV projects, which served as important sources of inspiration for the development of this platform.

---

# Contact

**Author**

[smxj@discente.ifpe.edu.br](mailto:smxj@discente.ifpe.edu.br)

**GitHub**

https://github.com/usuario/OpenPortableRetinograph
