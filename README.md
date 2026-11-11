# 🫁 PulmoLens – AI-Powered Pneumonia Detection & Explainability

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Keras%203-TensorFlow%202.16+-orange.svg)](https://keras.io/)
[![XAI](https://img.shields.io/badge/Explainability-Grad--CAM%20%7C%20LIME%20%7C%20SHAP-green.svg)](https://github.com/RanbeerReddy/PulmoLense)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **PulmoLens** is a production-grade, clinical-inspired web application designed for automated pneumonia classification and multi-modal feature explainability from chest radiographs (CXR).

---

## 🏥 Medical Disclaimer

> **⚠️ IMPORTANT NOTICE:**  
> This application is developed for educational and research purposes only. It is not a medical diagnostic device and should not replace professional medical evaluation. Predictions may be incorrect. Consult a qualified healthcare professional for medical advice.

---

## 🌟 Key Features

- **Multi-Model Inference Architecture:**
  - **Custom CNN (Selected Best Model):** A 4-stage convolutional network with batch normalization, dropout, and global average pooling, achieving **95.02% Accuracy** and **0.9877 ROC-AUC** on the independent test set.
  - **DenseNet121 (Transfer Learning):** Pretrained ImageNet feature extractor with fine-tuned classification head, achieving **98.44% Precision** and **95.71% Specificity**.
- **Adjustable Decision Thresholding:**
  - Interactive decision slider (0.10 to 0.90) with visual indicator for the validation-derived **Youden's J statistic optimal threshold (0.80)**.
- **Tri-Method Explainable AI (XAI) Suite:**
  - **Grad-CAM:** Near-instantaneous visual activation maps highlighting receptive fields in the final convolutional layers.
  - **LIME:** SLIC superpixel segmentation identifying positive and negative regional evidence.
  - **SHAP:** Partition SHAP with blur masking providing game-theoretic Shapley attributions.
- **Robust Clinical Usability:**
  - Zero external data dependency: Runs anywhere with pure inference.
  - Preloaded test radiographs (Normal, Bacterial Pneumonia, Viral Pneumonia) for instant demonstration.
  - Strict input validation and technical image metadata display.

---

## 📊 Verified Model Performance

All metrics are strictly verified against the independent **904-image test set** (233 NORMAL, 671 PNEUMONIA) with patient-level leakage prevention:

| Model Architecture | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | PR-AUC | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN (Best)** | **95.02%** | 96.72% | **96.57%** | 90.56% | **96.64%** | **0.9877** | **0.9957** | 1,489,665 |
| **DenseNet121 (TL)** | 94.25% | **98.44%** | 93.74% | **95.71%** | 96.03% | 0.9869 | 0.9954 | 7,175,745 |

### Confusion Matrix Breakdown (Test Set, N = 904)

- **Custom CNN:**
  - True Positives: **648** | True Negatives: **211**
  - False Positives: **22** | False Negatives: **23**
  - *Optimized for screening with high sensitivity (96.57%).*
- **DenseNet121:**
  - True Positives: **629** | True Negatives: **223**
  - False Positives: **10** | False Negatives: **42**
  - *Optimized for specificity (95.71%) and precision (98.44%).*

---

## 🗂️ Project Structure

```text
PulmoLens/
│
├── app.py                      # Production Streamlit application
│
├── models/                     # Saved Keras 3 model weights (<40 MB each)
│   ├── best_model.keras        # Top-performing Custom CNN
│   ├── custom_cnn_final.keras  # Final weights Custom CNN
│   └── densenet121_final.keras # Transfer learning DenseNet121
│
├── src/                        # Modular source package
│   ├── __init__.py
│   ├── preprocessing.py        # Image validation, resizing, normalization
│   ├── prediction.py           # Model catalog, loading, and inference logic
│   ├── explainability.py       # Grad-CAM, LIME, and SHAP implementations
│   └── utils.py                # Styling, chart generation, sample discovery
│
├── sample_images/              # Authentic test radiographs for 1-click evaluation
│   ├── normal_case_1.jpeg
│   ├── normal_case_2.jpeg
│   ├── pneumonia_bacterial.jpeg
│   └── pneumonia_viral.jpeg
│
├── tests/                      # Automated pytest unit & integration test suite
│   └── test_pipeline.py
│
├── .streamlit/                 # UI configuration and theme
│   └── config.toml
│
├── requirements.txt            # Production dependencies
├── packages.txt                # System libraries for Streamlit Community Cloud
├── .gitignore                  # Git hygiene rules
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/RanbeerReddy/PulmoLense.git
cd PulmoLense
```

### 2. Set Up Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Automated Tests

```bash
pytest tests/test_pipeline.py -v
```

### 4. Launch the Streamlit App Locally

```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

PulmoLens is fully configured for zero-friction deployment on **Streamlit Community Cloud**:

1. Fork or push this repository to your GitHub account (`https://github.com/RanbeerReddy/PulmoLense`).
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your repository: `RanbeerReddy/PulmoLense`.
5. Set Branch: `main`.
6. Set Main file path: `app.py`.
7. Click **"Deploy"**.

**Live Demo URL:** `https://pulmolense.streamlit.app/` (or custom subdomain assigned upon deployment).

---

## 🔬 Dataset & Data Leakage Prevention

- **Dataset:** Kaggle Pediatric Chest X-Ray cohort (Guangzhou Women and Children’s Medical Center).
- **Patient Leakage Prevention:** Standard random splits inadvertently place radiographs of the same patient into both training and testing subsets, artificially inflating accuracy. PulmoLens strictly parsed patient IDs using regex patterns (`personXX_`, `IM-XXXX`) to enforce **zero-patient overlap** across train (70%), validation (15%), and test (15%) partitions.

---

## 📜 License

This project is open-source and distributed under the [MIT License](LICENSE).

