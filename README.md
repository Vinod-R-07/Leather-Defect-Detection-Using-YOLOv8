# Leather Defect Detection Using YOLOv8 and OpenCV

An AI-powered leather defect detection system that automatically identifies leather surface defects using **YOLOv8**, **OpenCV**, and **Streamlit**. The project combines classical image preprocessing techniques with deep learning-based object detection to improve inspection accuracy and reduce manual quality checking.

---

# Project Overview

Leather quality inspection is an important process in the leather manufacturing industry. Traditional manual inspection is time-consuming, inconsistent, and highly dependent on human expertise. This project automates the inspection process by detecting defects directly from leather images.

The application first preprocesses the input image using OpenCV to enhance contrast and extract the leather region. The processed image is then passed to a trained YOLOv8 model, which detects different leather defects and displays them along with their confidence scores through a Streamlit web interface.

---

# Features

* Automatic leather defect detection using YOLOv8
* Leather region extraction using OpenCV
* Contrast enhancement using CLAHE
* Noise reduction using Gaussian Blur
* Adaptive image segmentation using Otsu Thresholding
* Automatic leather cropping and resizing
* Adjustable confidence threshold
* Boundary detection filtering to reduce false positives
* Interactive Streamlit web interface
* Displays detected defect names and confidence scores

---

# Dataset

This project uses the **Leather Defect Detection** dataset from **Roboflow Universe**.

**Dataset Link**

https://universe.roboflow.com/tharun-hd6k6/leather-defect-detection-liak8

### Defect Classes

* Crease
* Growth Marks
* Hole
* Rotten Surface
* Scratch
* Pinhole

The dataset is provided in YOLO format and divided into:

* Train
* Validation
* Test

---

# Detection Pipeline

```text
Input Leather Image
        │
        ▼
Convert RGB → Grayscale
        │
        ▼
Contrast Enhancement (CLAHE)
        │
        ▼
Gaussian Blur
        │
        ▼
Otsu Thresholding
        │
        ▼
Contour Detection
        │
        ▼
Leather Extraction
        │
        ▼
Automatic Cropping
        │
        ▼
Resize to 640 × 640
        │
        ▼
YOLOv8 Detection
        │
        ▼
Boundary Filtering
        │
        ▼
Final Detection Result
```

---

# Technologies Used

| Category             | Technology |
| -------------------- | ---------- |
| Programming Language | Python     |
| Object Detection     | YOLOv8     |
| Image Processing     | OpenCV     |
| Deep Learning        | PyTorch    |
| Web Application      | Streamlit  |
| Image Handling       | Pillow     |
| Numerical Computing  | NumPy      |

---

# Project Structure

```text
Leather-Defect-Detection-Using-YOLOv8/
│
├── app.py
├── detect.py
├── train.py
├── requirements.txt
├── dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
└── runs/
    └── detect/
        └── train-3/
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            └── training results
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Vinod-R-07/Leather-Defect-Detection-Using-YOLOv8.git
```

Go to the project directory:

```bash
cd Leather-Defect-Detection-Using-YOLOv8
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Upload a leather image and the application will automatically detect defects and display the results.

---

# Training the Model

To retrain the YOLOv8 model:

```bash
python train.py
```

The trained model is saved in:

```text
runs/detect/train-3/weights/best.pt
```

---

# Sample Results

The application displays:

* Original Image
* Leather Extraction + Contrast Enhancement
* Resized Image
* Final Detection Result
* Detected Defect Names
* Confidence Scores

---

# Future Scope

* Support additional leather defect categories
* Improve detection accuracy using larger datasets
* Deploy as a cloud-based inspection system
* Integrate real-time industrial camera support
* Export inspection reports automatically

---

# Author

**Vinod R**

Computer Science Engineering Student

2026
