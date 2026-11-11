"""
Preprocessing pipeline for PulmoLens chest X-ray analysis.
Conforms strictly to the notebook training protocol:
- Convert to 3-channel RGB (grayscale -> RGB)
- Resize to 224x224
- Normalize pixel values to [0.0, 1.0] float32
"""

from typing import Tuple, Dict, Any, Union
import numpy as np
from PIL import Image, UnidentifiedImageError
import io

TARGET_IMAGE_SIZE: Tuple[int, int] = (224, 224)


def validate_and_open_image(image_source: Union[str, bytes, io.BytesIO, Image.Image]) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Validates and opens an image from various input types.
    
    Args:
        image_source: File path, bytes, BytesIO, or PIL Image.
        
    Returns:
        Tuple of (PIL.Image, metadata_dict)
    
    Raises:
        ValueError: If image format is invalid or cannot be parsed.
    """
    try:
        if isinstance(image_source, Image.Image):
            pil_img = image_source
        elif isinstance(image_source, bytes):
            pil_img = Image.open(io.BytesIO(image_source))
        elif hasattr(image_source, 'read'):
            # Handles Streamlit UploadedFile or file-like objects
            image_source.seek(0)
            pil_img = Image.open(image_source)
        elif isinstance(image_source, str):
            pil_img = Image.open(image_source)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_source)}")
            
        pil_img.verify()
        
        # Reopen image because verify() clears the image data
        if hasattr(image_source, 'seek'):
            image_source.seek(0)
            pil_img = Image.open(image_source)
        elif isinstance(image_source, bytes):
            pil_img = Image.open(io.BytesIO(image_source))
        elif isinstance(image_source, str):
            pil_img = Image.open(image_source)
        else:
            pil_img = pil_img.copy()

    except (UnidentifiedImageError, OSError, Exception) as e:
        raise ValueError(f"Unable to read or decode image file: {str(e)}")

    metadata = {
        "original_width": pil_img.width,
        "original_height": pil_img.height,
        "original_mode": pil_img.mode,
        "original_format": pil_img.format or "UNKNOWN",
        "aspect_ratio": round(pil_img.width / max(1, pil_img.height), 3),
    }

    if pil_img.width < 32 or pil_img.height < 32:
        raise ValueError(
            f"Image dimensions ({pil_img.width}x{pil_img.height}) are too small for chest radiograph analysis."
        )

    return pil_img, metadata


def preprocess_image(
    image_source: Union[str, bytes, io.BytesIO, Image.Image],
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Preprocesses an input chest X-ray image according to the trained model specifications.
    
    Processing steps:
    1. Validate and open source image
    2. Convert to RGB (handles 1-channel grayscale, 4-channel RGBA, etc.)
    3. Resize to 224x224 using bilinear interpolation
    4. Normalize pixel values to range [0.0, 1.0] (float32)
    5. Expand batch dimensions to (1, 224, 224, 3)

    Returns:
        Tuple of:
          - batch_tensor: np.ndarray of shape (1, 224, 224, 3) with values in [0.0, 1.0]
          - display_array: np.ndarray of shape (224, 224, 3) for plotting and display
          - metadata: Dict with original image properties
    """
    pil_img, metadata = validate_and_open_image(image_source)
    
    # Standardize to RGB 3-channel
    rgb_img = pil_img.convert("RGB")
    
    # High-quality resize to target resolution
    resized_img = rgb_img.resize(target_size, Image.Resampling.BILINEAR)
    
    # Cast to float32 and scale to [0, 1]
    norm_arr = np.asarray(resized_img, dtype=np.float32) / 255.0
    
    # Ensure exact shape and value bounds
    norm_arr = np.clip(norm_arr, 0.0, 1.0)
    
    # Create batch tensor (1, 224, 224, 3)
    batch_tensor = np.expand_dims(norm_arr, axis=0)
    
    return batch_tensor, norm_arr, metadata

