"""
Explainable AI (XAI) module for PulmoLens.
Implements:
1. Grad-CAM (Gradient-Weighted Class Activation Mapping) for convolutional activation maps
2. LIME (Local Interpretable Model-Agnostic Explanations) with SLIC superpixel segmentation
3. SHAP (SHapley Additive exPlanations) with blur masking for pixel-level attribution
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

try:
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def generate_gradcam_heatmap(
    model: keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str = "conv2d_7"
) -> np.ndarray:
    """
    Computes a 2D Grad-CAM heatmap for a target convolutional layer.
    
    Args:
        model: Trained Keras model.
        img_array: Image tensor of shape (1, 224, 224, 3) normalized to [0, 1].
        last_conv_layer_name: Name of the target convolutional layer.
        
    Returns:
        2D numpy array of shape (224, 224) with values normalized in [0, 1].
    """
    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in the model.")

    grad_model = tf.keras.models.Model(
        model.inputs, [target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    # Global average pooling across feature map spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU on weighted activation map
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    heatmap_np = heatmap.numpy()
    
    # Resize heatmap to match image dimensions (224, 224)
    resized_heatmap = cv2.resize(heatmap_np, (img_array.shape[2], img_array.shape[1]))
    return np.clip(resized_heatmap, 0.0, 1.0)


def create_gradcam_overlay(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> np.ndarray:
    """
    Superimposes a Grad-CAM heatmap onto an RGB image.
    
    Args:
        original_img: 2D or 3D array in range [0, 1].
        heatmap: 2D array of shape (H, W) in range [0, 1].
        alpha: Overlay opacity.
        colormap_name: Matplotlib colormap name.
        
    Returns:
        Blended RGB image array in range [0, 1].
    """
    # Colorize heatmap (compatible with Matplotlib 3.7+)
    try:
        colormap = plt.colormaps[colormap_name]
    except (AttributeError, KeyError):
        colormap = cm.get_cmap(colormap_name)
    colored_heatmap = colormap(heatmap)[:, :, :3]
    
    if original_img.ndim == 2:
        original_rgb = np.stack([original_img] * 3, axis=-1)
    else:
        original_rgb = original_img

    overlay = alpha * colored_heatmap + (1.0 - alpha) * original_rgb
    return np.clip(overlay, 0.0, 1.0)


def run_lime_explanation(
    model: keras.Model,
    img_array: np.ndarray,
    num_samples: int = 500,
    n_segments: int = 60,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a LIME explanation for an input radiograph.
    
    Args:
        model: Trained model.
        img_array: 2D or 3D array of shape (224, 224, 3) in range [0, 1].
        num_samples: Number of perturbation samples.
        n_segments: SLIC superpixel count.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Tuple of:
          - weight_map: 2D float array of superpixel importance weights
          - boundary_img: RGB image with green boundaries around top supporting regions
          - mask: Boolean mask of top features
    """
    if not LIME_AVAILABLE:
        raise RuntimeError("LIME library is not installed.")

    def predict_fn(images: np.ndarray) -> np.ndarray:
        p = model.predict(np.asarray(images, dtype=np.float32), batch_size=32, verbose=0).ravel()
        return np.stack([1.0 - p, p], axis=1)

    seg_fn = SegmentationAlgorithm(
        "slic",
        n_segments=n_segments,
        compactness=10,
        sigma=1,
        start_label=1
    )
    
    explainer = lime_image.LimeImageExplainer(random_state=random_seed)
    explanation = explainer.explain_instance(
        img_array.astype("double"),
        predict_fn,
        labels=(1,),
        top_labels=None,
        hide_color=0,
        num_samples=num_samples,
        batch_size=32,
        segmentation_fn=seg_fn,
        random_seed=random_seed
    )

    weight_map = np.zeros(explanation.segments.shape, dtype=np.float32)
    for seg_id, weight in explanation.local_exp[1]:
        weight_map[explanation.segments == seg_id] = weight

    temp, mask = explanation.get_image_and_mask(
        1,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    boundary_img = mark_boundaries(img_array, mask, color=(0.0, 1.0, 0.0), mode="thick")
    return weight_map, boundary_img, mask


def run_shap_explanation(
    model: keras.Model,
    img_array: np.ndarray,
    max_evals: int = 300,
    random_seed: int = 42
) -> np.ndarray:
    """
    Generates a Partition SHAP attribution map for an input radiograph.
    
    Args:
        model: Trained model.
        img_array: Image array of shape (224, 224, 3) in range [0, 1].
        max_evals: Max function evaluations.
        random_seed: Random state.
        
    Returns:
        2D attribution map of shape (224, 224).
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError("SHAP library is not installed.")

    def predict_fn(images: np.ndarray) -> np.ndarray:
        # SHAP provides [0, 255] images, scale to [0, 1] for model
        scaled = np.asarray(images, dtype=np.float32) / 255.0
        p = model.predict(scaled, batch_size=32, verbose=0).ravel()
        return np.stack([1.0 - p, p], axis=1)

    masker = shap.maskers.Image("blur(64,64)", (224, 224, 3))
    explainer = shap.Explainer(
        predict_fn,
        masker,
        output_names=["NORMAL", "PNEUMONIA"]
    )
    
    input_uint8 = (img_array * 255.0)[None].astype(np.float32)
    shap_values = explainer(input_uint8, max_evals=max_evals, batch_size=32)
    
    vals = np.asarray(shap_values.values)[0]
    # Extract pneumonia class attribution
    if vals.ndim == 4:
        vals = vals[..., -1]
    
    attribution_map = vals.sum(axis=-1)
    return attribution_map
