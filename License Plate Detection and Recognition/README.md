# BSIT Final Year Project 2024

## LLM-Powered License Plate Detection and Recognition System

### 📌 Project Overview

This project presents an **automatic License Plate Detection and Recognition System** that combines **computer vision** and **large language/vision models** to identify vehicle license plates and extract text from them.

Traffic violations are a major issue worldwide. To enforce traffic laws and identify vehicles involved in violations, it is essential to **accurately detect vehicle license plates** and extract their numbers. This system automates the process by detecting vehicles, locating license plates, and recognizing the text on them.

The system uses **YOLOv8** for vehicle and license plate detection and a **Vision-Language Model (VLM)** called **Florence-2** developed by Microsoft to extract text from the detected license plate images.

By integrating these technologies, the project provides a **complete Automatic Number Plate Recognition (ANPR) system**.

---

## 🚀 Key Features

* Automatic **vehicle detection**
* **License plate detection** using deep learning
* **License plate tracking**
* **Text extraction (OCR) from license plates**
* Integration of **Vision-Language Models (VLM)**
* End-to-end **automatic number plate recognition system**

---

## 🧠 Technologies Used

* **Python**
* **YOLOv8 (Ultralytics)** – for vehicle and license plate detection
* **Florence-2 Vision Language Model** – for text extraction
* **NVIDIA NIM API** – for model inference
* **Hugging Face** – model hosting
* **OpenCV** – image processing

---

## ⚙️ System Architecture

1. **Vehicle Detection**

   * The system uses a **pre-trained YOLOv8 model** to detect vehicles in images or video streams.

2. **License Plate Detection**

   * YOLOv8 is also used to locate and crop the **license plate region** from the detected vehicle.

3. **Image Extraction**

   * The detected license plate is extracted and prepared for text recognition.

4. **Text Recognition**

   * The cropped license plate image is passed to the **Florence-2 Vision Language Model** through the **NVIDIA NIM API**.

5. **Output**

   * The model extracts the **license plate number as text**.

---

## 🔍 Florence-2 Model Overview

**Florence-2** is a **Vision-Language Model (VLM)** developed by Microsoft. It can perform multiple visual understanding tasks such as:

* Object Detection
* Optical Character Recognition (OCR)
* Visual Question Answering
* Image Captioning

Florence-2 models available on Hugging Face include:

| Model                       | Parameters               |
| --------------------------- | ------------------------ |
| Florence-2 Base             | Standard base model      |
| Florence-2 Base Fine-Tuned  | ~0.23 Billion parameters |
| Florence-2 Large            | Larger model             |
| Florence-2 Large Fine-Tuned | ~0.77 Billion parameters |

These models allow efficient **text extraction from images**, making them suitable for license plate recognition tasks.

---

## ☁️ NVIDIA NIM API

The **NVIDIA NIM framework** provides access to multiple AI models through APIs.

Features include:

* Access to **language models**
* **Vision-language models**
* **Multimodal AI models**
* **Stable diffusion models**

Developers can obtain an **API key** to perform inference using these models.

✔ The **first 100 API requests are free**, making it suitable for testing and development.

---

## 📂 Project Workflow

```
Input Image / Video
        │
        ▼
Vehicle Detection (YOLOv8)
        │
        ▼
License Plate Detection
        │
        ▼
License Plate Cropping
        │
        ▼
Florence-2 Model (via NVIDIA NIM API)
        │
        ▼
Text Extraction (License Plate Number)
        │
        ▼
Final Output
```

---

## 📊 Applications

* Traffic violation detection
* Smart city surveillance
* Automatic toll collection
* Parking management systems
* Law enforcement vehicle tracking

---

## 🎓 Academic Information

**Degree:** Bachelor of Science in Information Technology (BSIT)
**Project Type:** Final Year Project (FYP)
**Year:** 2024

---

## 📌 Future Improvements

* Real-time traffic camera integration
* Multi-camera vehicle tracking
* Database storage for detected plates
* Automatic fine generation system
* Improved OCR accuracy with custom training

---

## 🙌 Acknowledgment

This project was developed as part of the **BSIT Final Year Project 2024** to explore the integration of **computer vision and large language models for intelligent transportation systems**.