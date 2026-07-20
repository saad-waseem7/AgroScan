# ArgoScan: AI-Driven Crop Diagnosis and Treatment Recommendation

## Project Overview
ArgoScan is an end-to-end computer vision platform designed to identify plant diseases and provide targeted, sustainable mitigation advice from leaf images. By combining deep convolutional networks with an automated diagnostic workflow, it reduces the need for slow, manual inspection.

To improve safety and reliability, ArgoScan includes an image-quality validation gate that filters out poor-quality or non-leaf images before inference. It also provides chemical-free treatment suggestions, including biological controls, organic treatments, and long-term cultural practices, to support sustainable agriculture.

---

## Key Features
- Vision-based architecture: identifies crop diseases from RGB images with high precision.
- Dual-action pipeline: converts classification results into actionable treatment recommendations.
- Input quality gatekeeping: uses HSV masking and Canny edge analysis to validate image texture and composition.
- Production-grade modularity: separates data handling, image validation, inference, and presentation logic.
- Context-aware AI chat: provides strategic recommendations based on detected disease conditions.

---

## Directory Structure
The application layout is organized with clear separation of concerns:

```text
ArgoScan/
├── app.py
├── README.md
├── requirements.txt
├── assets/
│   └── styles.css
├── core/
│   ├── inference.py
│   └── vision.py
├── data/
│   └── treatments.json
├── docs/
│   └── ...
├── models/
│   ├── trained_model.h5
│   └── training_hist.json
```

---

## Dataset Specifications
This project uses the New Plant Disease Dataset, optimized for color image-based leaf diagnosis.

### Dataset Source
- Source: Kaggle
- Dataset name: Plant Disease Dataset (Color Images)
- Original dataset link: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- Download method: the dataset was downloaded manually from Kaggle and extracted locally.

### Dataset Characteristics
- Size: 87,900 RGB images.
- Scope: 38 disease and healthy classes across multiple crop species.
- Resolution pipeline: standardized input size of $128 \times 128 \times 3$ or $256 \times 256 \times 3$.
- Taxonomic breadth: includes common classes such as Apple Scab, Corn Common Rust, and Tomato Early/Late Blight.

### Dataset Structure
The dataset is pre-split into three main directories:
- Train: approximately 70,295 images across 38 class folders.
- Validation: approximately 17,572 images with the same 38-class structure.
- Test: 33 individual sample images used for final prediction checks.

> Note: The raw dataset is not included in this repository due to size limits. The original training subset can be downloaded from Kaggle under the New Plant Diseases Dataset.

---

## Installation and Setup

### Prerequisites
- Python 3.9 to 3.11

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
python app.py
```
