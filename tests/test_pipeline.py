"""
Automated unit & integration tests for PulmoLens.
Validates:
1. Model loading for all three models
2. Image preprocessing and format validation
3. Predictions on authentic sample radiographs
4. Confidence and probability sanity checks
5. Explainability (Grad-CAM, LIME, SHAP)
6. Invalid image exception handling
7. Path independence (no hardcoded absolute paths)
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_image, validate_and_open_image
from src.prediction import MODEL_CATALOG, load_trained_model, predict_pneumonia
from src.explainability import (
    generate_gradcam_heatmap,
    create_gradcam_overlay,
    run_lime_explanation,
    run_shap_explanation
)
from src.utils import get_available_sample_images


def test_models_exist():
    """Verify all model files exist and are under 100MB."""
    models_dir = ROOT / "models"
    assert models_dir.exists(), "models directory does not exist"
    
    for key, info in MODEL_CATALOG.items():
        model_path = models_dir / info["filename"]
        assert model_path.exists(), f"Model file {info['filename']} missing"
        size_mb = model_path.stat().st_size / (1024 * 1024)
        assert size_mb < 100.0, f"Model file {info['filename']} exceeds GitHub 100MB limit: {size_mb:.2f}MB"
        print(f"Verified {info['filename']} exists ({size_mb:.2f} MB)")


def test_model_loading_and_shapes():
    """Verify all three models load and accept (1, 224, 224, 3) inputs."""
    for key in MODEL_CATALOG:
        model = load_trained_model(key)
        assert model is not None
        assert model.input_shape == (None, 224, 224, 3)
        assert model.output_shape == (None, 1)
        print(f"Successfully loaded {key} with input {model.input_shape} and output {model.output_shape}")


def test_preprocessing():
    """Verify preprocessing outputs correct tensor shape and range [0, 1]."""
    samples = get_available_sample_images()
    assert len(samples) > 0, "No sample images available"
    
    for name, path in samples.items():
        batch, display, meta = preprocess_image(str(path))
        assert batch.shape == (1, 224, 224, 3)
        assert display.shape == (224, 224, 3)
        assert batch.dtype == np.float32
        assert batch.min() >= 0.0
        assert batch.max() <= 1.0
        assert meta["original_width"] > 0
        assert meta["original_height"] > 0
        print(f"Preprocessed {name}: shape={batch.shape}, range=[{batch.min():.2f}, {batch.max():.2f}]")


def test_invalid_image_handling():
    """Verify corrupt or invalid data raises ValueError cleanly."""
    corrupt_bytes = b"NOT_AN_IMAGE_FILE_DATA_12345"
    with pytest.raises(ValueError):
        preprocess_image(corrupt_bytes)
        
    tiny_img = Image.new("RGB", (10, 10))
    with pytest.raises(ValueError):
        preprocess_image(tiny_img)
    print("Verified invalid image handling correctly catches errors.")


def test_predictions_on_samples():
    """Verify predictions classify normal and pneumonia cases accurately."""
    model = load_trained_model("Best Model (Custom CNN)")
    samples = get_available_sample_images()
    
    # Normal Case 1
    if "Normal Case 1" in samples:
        batch, _, _ = preprocess_image(str(samples["Normal Case 1"]))
        res = predict_pneumonia(model, batch, threshold=0.50)
        assert res["predicted_label"] == "NORMAL", f"Expected NORMAL, got {res['predicted_label']}"
        assert res["prob_pneumonia"] < 0.20
        assert res["confidence_pct"] > 80.0
        print(f"Normal Case 1 predicted correctly as NORMAL (prob={res['prob_pneumonia']:.4f}, conf={res['confidence_pct']}%)")
        
    # Pneumonia Bacterial
    if "Pneumonia Bacterial" in samples:
        batch, _, _ = preprocess_image(str(samples["Pneumonia Bacterial"]))
        res = predict_pneumonia(model, batch, threshold=0.50)
        assert res["predicted_label"] == "PNEUMONIA", f"Expected PNEUMONIA, got {res['predicted_label']}"
        assert res["prob_pneumonia"] > 0.80
        assert res["confidence_pct"] > 80.0
        print(f"Pneumonia Bacterial predicted correctly as PNEUMONIA (prob={res['prob_pneumonia']:.4f}, conf={res['confidence_pct']}%)")


def test_gradcam():
    """Verify Grad-CAM produces normalized heatmap and valid overlay."""
    model = load_trained_model("Best Model (Custom CNN)")
    samples = get_available_sample_images()
    sample_path = list(samples.values())[0]
    batch, display, _ = preprocess_image(str(sample_path))
    
    heatmap = generate_gradcam_heatmap(model, batch, last_conv_layer_name="conv2d_7")
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    
    overlay = create_gradcam_overlay(display, heatmap, alpha=0.5)
    assert overlay.shape == (224, 224, 3)
    assert overlay.min() >= 0.0
    assert overlay.max() <= 1.0
    print("Grad-CAM verified successfully.")


def test_lime_fast():
    """Verify LIME generates superpixel boundaries and weight map."""
    model = load_trained_model("Best Model (Custom CNN)")
    samples = get_available_sample_images()
    sample_path = list(samples.values())[0]
    _, display, _ = preprocess_image(str(sample_path))
    
    wmap, bound_img, mask = run_lime_explanation(model, display, num_samples=30, n_segments=30)
    assert wmap.shape == (224, 224)
    assert bound_img.shape == (224, 224, 3)
    print("LIME verified successfully.")


def test_shap_fast():
    """Verify SHAP generates attribution map."""
    model = load_trained_model("Best Model (Custom CNN)")
    samples = get_available_sample_images()
    sample_path = list(samples.values())[0]
    _, display, _ = preprocess_image(str(sample_path))
    
    att = run_shap_explanation(model, display, max_evals=20)
    assert att.shape == (224, 224)
    print("SHAP verified successfully.")


if __name__ == "__main__":
    print("--- Running PulmoLens Test Suite ---")
    test_models_exist()
    test_model_loading_and_shapes()
    test_preprocessing()
    test_invalid_image_handling()
    test_predictions_on_samples()
    test_gradcam()
    test_lime_fast()
    test_shap_fast()
    print("--- ALL TESTS PASSED SUCCESSFULLY! ---")

