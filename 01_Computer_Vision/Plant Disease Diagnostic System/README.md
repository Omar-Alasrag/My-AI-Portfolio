# 🌿 Precision Agriculture: Plant Disease Diagnostic System

An AI-driven diagnostic tool that detects and localizes crop diseases using state-of-the-art Object Detection architectures.

## 🏗️ Model Architecture
* **Framework:** Faster R-CNN with a **MobileNetV3-Large** backbone (FPN).
* **Optimization:** Targeted at edge deployment—balancing mean Average Precision (mAP) with a lightweight backbone for real-time field use.
* **Custom Dataset Loader:** Built a robust PyTorch `Dataset` class to handle YOLO-format labels and convert them to COCO-style bounding boxes (xyxy).

## 📈 Metrics & Performance
* **Augmentation:** Uses `RandomAutocontrast` to handle varying lighting conditions typical in outdoor agricultural environments.
* **Evaluation:** Tracks **mAP@50** to ensure high localization accuracy of small disease spots on leaves.

## 🚀 How to Run
1. Organize your dataset in `data/plant_diseases_dataset` (with `/images` and `/labels`).
2. Run `python plant_disease_detection.py`.