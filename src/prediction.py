"""
Model loading and inference module for PulmoLens.
Integrates the trained Keras models with verified performance metrics from the experimental notebook.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import keras
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Verified evaluation metrics strictly derived from the test set in Pneumonia_XAI_MiniProject.ipynb
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "Best Model (Custom CNN)": {
        "filename": "best_model.keras",
        "display_name": "Best Model (Custom CNN)",
        "architecture": "4-Block Deep Convolutional Neural Network",
        "description": "Selected as the top-performing model based on validation ROC-AUC. Features 4 cascaded Conv-BatchNorm-ReLU blocks with dropout regularization and Global Average Pooling.",
        "params": "1,489,665",
        "metrics": {
            "Accuracy": 0.9502,
            "Precision": 0.9672,
            "Recall (Sensitivity)": 0.9657,
            "Specificity": 0.9056,
            "F1-Score": 0.9664,
            "ROC-AUC": 0.9877,
            "PR-AUC": 0.9957,
        },
        "optimal_threshold": 0.80,  # Youden's J statistic threshold from validation analysis
        "default_threshold": 0.50,
        "supports_gradcam": True,
        "gradcam_layer": "conv2d_7",
    },
    "Custom CNN": {
        "filename": "custom_cnn_final.keras",
        "display_name": "Custom CNN (Final Weights)",
        "architecture": "4-Block Deep Convolutional Neural Network",
        "description": "Custom-designed 4-stage convolutional architecture trained with Adam optimizer and binary cross-entropy on patient-split radiographs.",
        "params": "1,489,665",
        "metrics": {
            "Accuracy": 0.9502,
            "Precision": 0.9672,
            "Recall (Sensitivity)": 0.9657,
            "Specificity": 0.9056,
            "F1-Score": 0.9664,
            "ROC-AUC": 0.9877,
            "PR-AUC": 0.9957,
        },
        "optimal_threshold": 0.80,
        "default_threshold": 0.50,
        "supports_gradcam": True,
        "gradcam_layer": "conv2d_7",
    },
    "DenseNet121": {
        "filename": "densenet121_final.keras",
        "display_name": "DenseNet121 (Transfer Learning)",
        "architecture": "DenseNet121 Pretrained Backbone + Fine-Tuned Dense Head",
        "description": "Transfer learning using ImageNet pretrained DenseNet121 with internal normalization layer, dense connectivity, and custom classification head.",
        "params": "7,175,745",
        "metrics": {
            "Accuracy": 0.9425,
            "Precision": 0.9844,
            "Recall (Sensitivity)": 0.9374,
            "Specificity": 0.9571,
            "F1-Score": 0.9603,
            "ROC-AUC": 0.9869,
            "PR-AUC": 0.9954,
        },
        "optimal_threshold": 0.50,
        "default_threshold": 0.50,
        "supports_gradcam": False,
        "gradcam_layer": None,
    },
}


@st.cache_resource(show_spinner=False)
def load_trained_model(model_key: str) -> keras.Model:
    """
    Loads and caches a Keras model by its catalog key.
    
    Args:
        model_key: Key in MODEL_CATALOG.
        
    Returns:
        Loaded Keras Functional model.
        
    Raises:
        FileNotFoundError: If model file does not exist on disk.
    """
    if model_key not in MODEL_CATALOG:
        raise KeyError(f"Unknown model key '{model_key}'. Available: {list(MODEL_CATALOG.keys())}")
        
    info = MODEL_CATALOG[model_key]
    model_path = MODELS_DIR / info["filename"]
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights file '{info['filename']}' not found in {MODELS_DIR}."
        )
        
    # Compile=False since we use model purely for inference
    model = keras.models.load_model(str(model_path), compile=False)
    return model


def predict_pneumonia(
    model: keras.Model,
    input_tensor: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Executes binary classification inference on a preprocessed chest radiograph.
    
    Args:
        model: Loaded Keras model.
        input_tensor: Tensor of shape (1, 224, 224, 3) with float values in [0.0, 1.0].
        threshold: Decision boundary threshold for the PNEUMONIA class (default: 0.50).
        
    Returns:
        Dict containing:
          - predicted_label: "PNEUMONIA" or "NORMAL"
          - prob_pneumonia: Float [0.0, 1.0]
          - prob_normal: Float [0.0, 1.0]
          - confidence_pct: Float [0.0, 100.0]
          - threshold_used: Float
          - risk_category: "Low Risk", "Moderate Risk", or "High Risk"
    """
    raw_pred = model.predict(input_tensor, verbose=0)
    prob_pneumonia = float(np.ravel(raw_pred)[0])
    # Clamp bounds to avoid numerical edge artifacts
    prob_pneumonia = max(0.0, min(1.0, prob_pneumonia))
    prob_normal = 1.0 - prob_pneumonia
    
    if prob_pneumonia >= threshold:
        predicted_label = "PNEUMONIA"
        confidence_pct = prob_pneumonia * 100.0
    else:
        predicted_label = "NORMAL"
        confidence_pct = prob_normal * 100.0

    # Clinical risk stratification
    if prob_pneumonia >= 0.75:
        risk_category = "High Probability of Pneumonia"
        severity_color = "#E53E3E"  # Red
    elif prob_pneumonia >= 0.40:
        risk_category = "Indeterminate / Moderate Concern"
        severity_color = "#DD6B20"  # Orange
    else:
        risk_category = "Consistent with Normal Chest Radiograph"
        severity_color = "#38A169"  # Green

    return {
        "predicted_label": predicted_label,
        "prob_pneumonia": prob_pneumonia,
        "prob_normal": prob_normal,
        "confidence_pct": round(confidence_pct, 2),
        "threshold_used": threshold,
        "risk_category": risk_category,
        "severity_color": severity_color,
    }

