# Drawings

This directory contains the assembly drawings used for the construction of the portable retinograph.

The documents provided serve as references for mechanical assembly, optical component positioning, and embedded hardware integration.

---

# Files

## Componentes.pdf

Exploded-view drawing of the retinograph showing all components used in the assembly.

The main components are identified as follows:

| Identifier | Description                    |
| ---------- | ------------------------------ |
| A          | Facial support (TPU)           |
| B          | Optical housing                |
| C          | Main enclosure                 |
| D          | Smartphone mount               |
| E          | LCD display adapter (optional) |
| F          | First-surface mirrors          |
| G          | Mirror holder                  |
| H          | Condensing lens holder         |
| I          | Raspberry Pi 4                 |
| J          | M12 camera module              |
| K          | Smartphone (Web interface)     |
| L          | +20D condensing lens           |
| M          | Electronic circuit board       |
| N          | LED illumination system        |

---

## Esquema.pdf

Dimensional drawing of the prototype.

It includes:

* External dimensions
* Subassembly layout
* Optical alignment
* Camera position
* Condensing lens position
* Overall device dimensions

Approximate prototype dimensions:

* **Length:** 26 cm
* **Height:** 17 cm
* **Width:** 11 cm

---

# Assembly Sequence

The recommended assembly procedure is as follows:

1. Print the PLA and TPU components.
2. Fabricate the MDF structural base.
3. Assemble the optical housing.
4. Install the first-surface mirrors.
5. Install the condensing lens holder.
6. Mount the +20D condensing lens.
7. Install the M12 camera module.
8. Insert the optical housing into the main enclosure.
9. Adjust the focusing mechanism.
10. Install the LED illumination system.
11. Secure the Raspberry Pi 4.
12. Connect the camera, LEDs, and Raspberry Pi.
13. Install the electronic circuit board.
14. Connect the power supply.
15. Perform optical alignment tests.
16. Fine-tune the focus.
17. Establish the Wi-Fi connection using a smartphone.
18. Perform final image acquisition tests.

---

# Notes

* The facial support is manufactured from TPU to provide improved comfort and accommodate different facial anatomies.
* The system was designed to use a smartphone as the primary user interface, reducing the overall cost of the device.
* The architecture also supports the installation of a 5-inch or 3.5-inch LCD display, although this configuration is optional.
* All structural components were designed for additive manufacturing using 3D printing, enabling straightforward reproduction and maintenance of the device.

---

# Related Documentation

See also:

* `/hardware/CAD`
* `/hardware/STL`
* `/hardware/BOM`
* `/hardware/Schematics`
* `/hardware/Datasheets`
