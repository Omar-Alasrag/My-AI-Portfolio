# AI & Machine Learning Portfolio

This repository contains a collection of my AI, machine learning, and deep learning projects** that I built

The focus of this portfolio is:
- Understanding models from first principles
- Implementing ideas correctly
- Comparing different algorithms using evaluation metrics
- Combining classical ML with modern deep learning and transformers

---
    
## Skills & Topics Covered

- **CNNs & Computer Vision** → CIFAR-10, SSD Object Detection, Face Recognition  
- **Transformers (from scratch)** → Custom Transformer Seq2Seq (PyTorch)  
- **NLP Fine-tuning** → NER, Textual Entailment (RTE), Sentiment Analysis, Translation  
- **Classical ML** → Regression, Classification, Clustering, Association Rules  
- **Optimization & Evaluation** → Grid Search, K-Fold, ROC-AUC, BLEU  

---

## Project Overview

### 01. Computer Vision
#### Deep Learning & Real-Time Applications
- **Real-Time Vehicle Counting using YOLO and ByteTrack** – Multi-object tracking for real-time traffic monitoring.
- **Automatic License Plate Recognition (ALPR) System** – YOLO-based detection + OCR text extraction.
- **Real-Time Vehicle Counting using YOLO and ByteTrack** – Multi-object tracking for real-time traffic monitoring.
- **Automatic License Plate Recognition (ALPR) System** – YOLO-based detection + OCR text extraction
- **Plant Disease Diagnostic System** – Real-time Detection with Faster R-CNN / SSDLite
- **Face Mask Detection using MobileNetV3** – Transfer learning for real-time mask classification.
- **CIFAR-10 CNN Image Classifier with Data Augmentation** – Multi-class image classification using CNNs.
- **SSD Object Detection** – Single Shot Detector for general object detection tasks.
#### Face & Pose Estimation
- **Face Recognition using OpenCV** – Face identification and verification using classical feature extraction.
- **Face Distance Estimation using Facial Landmarks** – Distance measurement between facial keypoints.
- **Pose-Based Exercise Repetition Counter** – Keypoint detection to count exercise repetitions.
#### Classical Computer Vision / Feature Engineering
- **Feature Matching: ORB vs SIFT** – Comparing classical feature detectors for image matching.
- **Motion Detection using Frame Differencing** – Real-time motion tracking in video frames.
- **Vehicle Counting System** – Counting vehicles using classical CV approaches.

---

### 02. NLP & Transformers

#### Transformers

**From-Scratch Implementation**
- Custom Transformer Seq2Seq model including:
  - Multi-Head Attention
  - Positional Encoding
  - Padding and causal masking
  - Residual connections and layer normalization

**Fine-Tuned Transformer Models**
- Named Entity Recognition (DistilBERT)
- Textual Entailment on GLUE (RTE)
- Neural Machine Translation evaluated with BLEU
- Twitter sentiment analysis evaluated with F1-score and ROC-AUC

---

### 03. Generative AI
- DCGAN trained on CIFAR-10 for image generation with Pytorch & Tensorflow

---

### 04. Recommender Systems
- AutoEncoder-based collaborative filtering using MovieLens data

---

### 05. Predictive Analytics & Classical ML
- Regression and classification model comparisons
- Customer segmentation using clustering techniques
- Market basket analysis with Apriori
- Thompson Sampling for ad selection optimization
- Hyperparameter tuning with Grid Search and K-Fold cross-validation
- ANN-based customer churn prediction

---

## Tools & Libraries
- Python, PyTorch, Hugging Face Transformers, Tensorflow
- Scikit-learn, NumPy, Pandas, Matplotlib, OpenCV
- Evaluation metrics: Accuracy, F1-score, ROC-AUC, BLEU
- LangChain, ChromaDB, Google Gemini API, Tavily
- asyncio, Dotenv

---

### 06. LLM Orchestration & RAG
- **Autonomous Doc Assistant** – RAG system for crawling and querying technical documentation.
- **Async Ingestion** – High-concurrency indexing using asyncio semaphores and ChromaDB.
- **Web Crawling** – Automated site-to-vector pipeline using TavilyCrawl and RecursiveCharacterTextSplitter.
- **Tool-Calling Agent** – Gemini 2.0 Flash Lite agent with strict retrieval-only system prompts and source attribution.
- **History Management** – Custom filtering of ToolMessages to optimize context window for multi-turn chat.
