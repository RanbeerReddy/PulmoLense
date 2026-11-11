"""
Utility functions, plotting helpers, and UI styling for PulmoLens.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# Exact color codes from the research notebook
CLASS_COLORS = {
    "NORMAL": "#2E7D32",     # Forest Green
    "PNEUMONIA": "#C62828",   # Crimson Red
    "ACCENT": "#00838F",     # Deep Cyan
    "MUTED": "#78909C"       # Blue-grey
}

MEDICAL_DISCLAIMER = (
    "This application is developed for educational and research purposes only. "
    "It is not a medical diagnostic device and should not replace professional medical evaluation. "
    "Predictions may be incorrect. Consult a qualified healthcare professional for medical advice."
)

SAMPLE_IMAGES_DIR = Path(__file__).resolve().parent.parent / "sample_images"


def get_available_sample_images() -> Dict[str, Path]:
    """
    Returns available sample chest X-ray images.
    """
    if not SAMPLE_IMAGES_DIR.exists():
        return {}
    
    samples = {}
    for p in sorted(SAMPLE_IMAGES_DIR.glob("*.jpeg")) + sorted(SAMPLE_IMAGES_DIR.glob("*.jpg")) + sorted(SAMPLE_IMAGES_DIR.glob("*.png")):
        name_clean = p.stem.replace("_", " ").title()
        samples[name_clean] = p
    return samples


def create_probability_bar_chart(prob_pneumonia: float, prob_normal: float) -> plt.Figure:
    """
    Creates a clean, publication-ready horizontal probability bar chart.
    """
    fig, ax = plt.subplots(figsize=(6, 1.8), dpi=150)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1E222B')

    categories = ["NORMAL", "PNEUMONIA"]
    probs = [prob_normal, prob_pneumonia]
    colors = [CLASS_COLORS["NORMAL"], CLASS_COLORS["PNEUMONIA"]]

    bars = ax.barh(categories, probs, color=colors, height=0.55, edgecolor="none")
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="#B0BEC5", fontsize=9)
    ax.set_yticklabels(categories, color="#ECEFF1", fontsize=10, fontweight="bold")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#37474F")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, bottom=True, colors="#B0BEC5")
    ax.grid(axis="x", linestyle="--", alpha=0.2, color="#FFFFFF")

    # Add text labels on the bars
    for bar, p in zip(bars, probs):
        w = bar.get_width()
        pos = max(w + 0.02, 0.05) if w < 0.85 else w - 0.12
        text_color = "#FFFFFF" if w >= 0.85 else "#FFFFFF"
        ax.text(
            pos,
            bar.get_y() + bar.get_height() / 2,
            f"{p * 100:.1f}%",
            va="center",
            ha="left" if w < 0.85 else "right",
            color=text_color,
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    return fig


def create_custom_css() -> str:
    """
    Returns custom CSS for a modern, sleek, high-end medical UI.
    """
    return """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }
    
    .status-badge-pneumonia {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        color: #FCA5A5;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
    }
    
    .status-badge-normal {
        display: inline-block;
        background-color: rgba(34, 197, 94, 0.15);
        border: 1px solid #22C55E;
        color: #86EFAC;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
    }
    
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 0.85rem 1.2rem;
        margin: 1.2rem 0;
        color: #FDE68A;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .header-subtext {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Clean tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
    }
    </style>
    """

