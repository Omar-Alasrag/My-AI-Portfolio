# 🚗 Automatic License Plate Recognition (ALPR) System

A robust end-to-end pipeline for detecting vehicle license plates and extracting alphanumeric data using deep learning and OCR.

## 🛠️ Technical Workflow
* **Detection:** Utilizes **YOLOv11** (the latest Ultralytics architecture) fine-tuned on a custom License Plate dataset.
* **Spatial Logic:** Implemented a **Point Polygon Test** to define a specific "Detection Zone," reducing false positives from background traffic.
* **OCR Engine:** Uses **Tesseract OCR** with a optimized Page Segmentation Mode (`--psm 7`) for sparse text extraction.
* **Data Logging:** Automatically exports detected plate numbers with high-resolution timestamps to a structured `.txt` database.

## 📊 Key Results
* **Real-time Performance:** Optimized frame-skipping logic (processing every 3rd frame) to maintain smooth inference on CPU/GPU.
* **Robustness:** Includes character cleaning via string manipulation to handle common OCR noise like brackets and commas.

## 🚀 How to Run
1. Install Tesseract-OCR on your system.
2. Update the `tesseract_cmd` path in `license_plate_recognition.py`.
3. Run `python license_plate_recognition.py`.