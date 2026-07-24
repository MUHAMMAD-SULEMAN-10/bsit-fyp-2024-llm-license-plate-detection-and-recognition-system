# LLM-Powered License Plate Detection and Recognition System

## Professional Project Information

| Field          | Details                                                    |
| -------------- | ---------------------------------------------------------- |
| Project Title  | LLM-Powered License Plate Detection and Recognition System |
| Degree Program | Bachelor of Science in Information Technology (BSIT)       |
| Project Type   | Final Year Project (FYP)                                   |
| Department     | Department of Computer Science                             |
| Academic Year  | 2024                                                       |
| Project Domain | Computer Vision & Artificial Intelligence                  |
| Supervisor     | Usman Saif                                                 |
| Team Size      | 4 Members                                                  |
| Group Leader   | Muhammad Suleman (Roll No: 032303)                        |

---

# Team Members & Responsibilities

| Team Member                | Role                            | Responsibilities                                                                                                                                         |
| -------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Muhammad Suleman (032303) | Group Leader & AI Engineer      | Led the project development, worked on model building, dataset preparation, YOLOv8 implementation, model training, model integration, and system testing |
| Usman Asgher   (032315)            | Backend & Integration Developer | Assisted in backend development, API integration, data handling, and connecting model outputs with the application workflow                              |
| Uzair Ahmad    (0323088)            | Research & Documentation Lead   | Worked on technical research, project documentation, literature review, workflow design, and presentation preparation                                    |
| Hassan Ijaz    (032316)          | Frontend & Testing Support      | Assisted in interface design, result visualization, testing, debugging, and overall project support                                                      |
| Usman Saif                 | Project Supervisor              | Provided academic supervision, project guidance, technical feedback, and research direction throughout the project lifecycle                             |

---

#  Project Overview

This project presents an automatic License Plate Detection and Recognition System that combines computer vision and large language/vision models to identify vehicle license plates and extract text from them.

Traffic violations are a major issue worldwide. To enforce traffic laws and identify vehicles involved in violations, it is essential to accurately detect vehicle license plates and extract their numbers. This system automates the process by detecting vehicles, locating license plates, and recognizing the text on them.

The system uses YOLOv8 for vehicle and license plate detection and a Vision-Language Model (VLM) called Florence-2 developed by Microsoft to extract text from the detected license plate images.

By integrating these technologies, the project provides a complete Automatic Number Plate Recognition (ANPR) system.

---

#  Key Features

* Automatic vehicle detection
* License plate detection using deep learning
* License plate tracking
* Text extraction (OCR) from license plates
* Integration of Vision-Language Models (VLM)
* End-to-end automatic number plate recognition system

---

#  Technologies Used

| Technology                       | Purpose                             |
| -------------------------------- | ----------------------------------- |
| Python                           | Core programming language           |
| YOLOv8 (Ultralytics)             | Vehicle and license plate detection |
| Florence-2 Vision Language Model | OCR and text extraction             |
| NVIDIA NIM API                   | Model inference and API access      |
| Hugging Face                     | Model hosting and deployment        |
| OpenCV                           | Image and video processing          |

---

#  System Architecture

## 1. Vehicle Detection

The system uses a pre-trained YOLOv8 model to detect vehicles in images or video streams.

## 2. License Plate Detection

YOLOv8 is used to locate and crop the license plate region from detected vehicles.

## 3. Image Extraction

The detected license plate region is extracted and prepared for text recognition.

## 4. Text Recognition

The cropped license plate image is passed to the Florence-2 Vision Language Model through the NVIDIA NIM API.

## 5. Final Output

The system extracts and displays the license plate number as text.

---

#  Project Workflow

```text
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

#  Florence-2 Model Overview

Florence-2 is a Vision-Language Model (VLM) developed by Microsoft. It supports multiple visual understanding tasks such as:

* Object Detection
* Optical Character Recognition (OCR)
* Visual Question Answering
* Image Captioning

## Florence-2 Models

| Model                       | Description              |
| --------------------------- | ------------------------ |
| Florence-2 Base             | Standard base model      |
| Florence-2 Base Fine-Tuned  | ~0.23 Billion parameters |
| Florence-2 Large            | Larger model             |
| Florence-2 Large Fine-Tuned | ~0.77 Billion parameters |

These models provide efficient OCR and text extraction capabilities for license plate recognition systems.

---

# ☁️ NVIDIA NIM API

The NVIDIA NIM framework provides access to multiple AI and multimodal models through APIs.

## Features

* Access to large language models
* Vision-language model integration
* Multimodal AI support
* Scalable inference APIs

The API enables efficient model inference for license plate text recognition.

---

#  Applications

* Traffic violation detection
* Smart city surveillance
* Automatic toll collection
* Parking management systems
* Law enforcement vehicle tracking

---

#  Future Improvements

* Real-time traffic camera integration
* Multi-camera vehicle tracking
* Database storage for detected plates
* Automatic fine generation system
* Improved OCR accuracy with custom training
* Deployment as a real-time web application

---

#  Academic Contribution

This project was developed as part of the BSIT Final Year Project (FYP) 2024 to explore the integration of computer vision, deep learning, and large vision-language models for intelligent transportation systems.

The project demonstrates practical implementation of AI-powered Automatic Number Plate Recognition (ANPR) systems using modern deep learning technologies.

---

# 🙌 Acknowledgment

Special thanks to our respected supervisor, Usman Saif, Department of Computer Science, for his continuous guidance, valuable feedback, and support throughout the development of this Final Year Project.
