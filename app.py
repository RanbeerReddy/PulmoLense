"""
PulmoLens – AI-Powered Pneumonia Detection & Explainability
Production Streamlit Application
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Add parent directory to path for module resolution
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing import preprocess_image
from src.prediction import MODEL_CATALOG, load_trained_model, predict_pneumonia
from src.explainability import (
    generate_gradcam_heatmap,
    create_gradcam_overlay,
    run_lime_explanation,
    run_shap_explanation,
    LIME_AVAILABLE,
    SHAP_AVAILABLE
)
from src.utils import (
    MEDICAL_DISCLAIMER,
    CLASS_COLORS,
    get_available_sample_images,
    create_probability_bar_chart,
    create_custom_css
)

# Page configuration
st.set_page_config(
    page_title="PulmoLens – AI-Powered Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS styling
st.markdown(create_custom_css(), unsafe_allow_html=True)


def render_header():
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("🫁 PulmoLens")
        st.markdown(
            "<p class='header-subtext'>Clinical-Grade Deep Learning & Explainable AI for Chest Radiograph Analysis</p>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 10px;">
                <span style="background: rgba(2, 132, 199, 0.2); color: #38BDF8; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600; border: 1px solid rgba(56, 189, 248, 0.4);">
                    v1.0 Production
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="disclaimer-box">
            <strong>⚠️ Medical Disclaimer:</strong> {MEDICAL_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    render_header()

    # Sidebar controls
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/lungs.png", width=64)
        st.header("⚙️ Configuration")
        
        # Model Selection
        selected_model_name = st.selectbox(
            "Select Deep Learning Model",
            options=list(MODEL_CATALOG.keys()),
            index=0,
            help="Choose between the verified Custom CNN and DenseNet121 transfer learning architectures."
        )
        model_info = MODEL_CATALOG[selected_model_name]
        
        st.caption(f"**Architecture:** {model_info['architecture']}")
        st.caption(f"**Parameters:** {model_info['params']}")
        st.caption(f"**Test ROC-AUC:** {model_info['metrics']['ROC-AUC']:.4f}")

        st.divider()

        # Decision threshold slider
        st.subheader("🎯 Decision Threshold")
        threshold = st.slider(
            "Pneumonia Classification Cutoff",
            min_value=0.10,
            max_value=0.90,
            value=float(model_info["default_threshold"]),
            step=0.05,
            help=f"Default: 0.50. Optimal Youden's J statistic threshold from notebook validation: {model_info['optimal_threshold']:.2f}."
        )
        if abs(threshold - model_info["optimal_threshold"]) < 0.01:
            st.info(f"✨ Using Youden's optimal validation threshold ({model_info['optimal_threshold']:.2f})")

        st.divider()

        # Sample quick loader
        st.subheader("📁 Preloaded Test Cases")
        samples = get_available_sample_images()
        selected_sample = None
        if samples:
            sample_choice = st.selectbox(
                "Quick-test with verified sample X-rays:",
                options=["-- None (Upload your own) --"] + list(samples.keys()),
                index=0
            )
            if sample_choice != "-- None (Upload your own) --":
                selected_sample = samples[sample_choice]

        st.divider()
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #94A3B8;">
                Developed with TensorFlow, Keras 3, and Streamlit.<br>
                Trained on Guangzhou Women & Children's Medical Center dataset.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Main Tabs
    tab_diagnose, tab_performance, tab_about = st.tabs([
        "🔬 Diagnosis & Explainability",
        "📊 Model Benchmarks & Metrics",
        "📖 Clinical Context & Methodology"
    ])

    with tab_diagnose:
        col_input, col_results = st.columns([1, 1.2], gap="large")

        active_image_source = None

        with col_input:
            st.subheader("1. Chest X-Ray Input")
            uploaded_file = st.file_uploader(
                "Upload a chest radiograph (Anteroposterior or Posteroanterior view)",
                type=["jpg", "jpeg", "png"],
                help="Accepts high-resolution JPG or PNG chest radiographs."
            )

            if uploaded_file is not None:
                active_image_source = uploaded_file
            elif selected_sample is not None:
                active_image_source = str(selected_sample)
                st.caption(f"Active sample: **{Path(selected_sample).stem.replace('_', ' ').title()}**")

            if active_image_source is None:
                st.info("👆 Please upload a chest X-ray image or select a sample case from the sidebar to begin analysis.")
                # Show an illustrated placeholder
                st.markdown(
                    """
                    <div style="border: 2px dashed rgba(148, 163, 184, 0.25); border-radius: 12px; padding: 3rem 1.5rem; text-align: center; color: #94A3B8;">
                        <span style="font-size: 3rem;">🩻</span><br><br>
                        <strong>No chest radiograph loaded</strong><br>
                        Supports DICOM-derived JPEG, JPG, and PNG formats.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if active_image_source is not None:
            # Step 1: Preprocess Image
            try:
                batch_tensor, display_arr, meta = preprocess_image(active_image_source)
            except Exception as e:
                with col_input:
                    st.error(f"❌ Image Processing Error: {str(e)}")
                return

            with col_input:
                st.image(
                    display_arr,
                    caption=f"Input Radiograph (Resized to 224x224 RGB | Original: {meta['original_width']}x{meta['original_height']} {meta['original_mode']})",
                    use_container_width=True
                )
                
                with st.expander("🔍 Image Technical Metadata", expanded=False):
                    st.write(f"- **Original Dimensions:** {meta['original_width']} × {meta['original_height']} px")
                    st.write(f"- **Original Color Mode:** {meta['original_mode']}")
                    st.write(f"- **Original File Format:** {meta['original_format']}")
                    st.write(f"- **Aspect Ratio:** {meta['aspect_ratio']}")
                    st.write("- **Model Input Tensor:** (1, 224, 224, 3) normalized to [0.0, 1.0]")

            # Step 2: Prediction
            with col_results:
                st.subheader("2. AI Diagnostic Assessment")
                with st.spinner("Analyzing radiograph with deep neural network..."):
                    try:
                        model = load_trained_model(selected_model_name)
                        result = predict_pneumonia(model, batch_tensor, threshold=threshold)
                    except Exception as e:
                        st.error(f"Model Inference Error: {str(e)}")
                        return

                # Display Results Card
                pred_label = result["predicted_label"]
                prob_pneumonia = result["prob_pneumonia"]
                prob_normal = result["prob_normal"]
                confidence = result["confidence_pct"]

                badge_class = "status-badge-pneumonia" if pred_label == "PNEUMONIA" else "status-badge-normal"
                badge_icon = "🚨" if pred_label == "PNEUMONIA" else "✅"

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                            <span class="{badge_class}">{badge_icon} {pred_label}</span>
                            <span style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC;">
                                {confidence:.1f}% Confidence
                            </span>
                        </div>
                        <div style="font-size: 0.95rem; color: #CBD5E1; margin-bottom: 0.5rem;">
                            <strong>Risk Classification:</strong> <span style="color: {result['severity_color']}; font-weight: 600;">{result['risk_category']}</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #94A3B8;">
                            Model: <em>{selected_model_name}</em> | Decision Threshold: <strong>{threshold:.2f}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Probability Chart
                st.markdown("##### Prediction Probability Distribution")
                fig_bar = create_probability_bar_chart(prob_pneumonia, prob_normal)
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)

                c_metric1, c_metric2 = st.columns(2)
                with c_metric1:
                    st.metric(
                        label="P(PNEUMONIA)",
                        value=f"{prob_pneumonia * 100:.2f}%",
                        delta=f"{(prob_pneumonia - threshold) * 100:+.1f}% vs threshold"
                    )
                with c_metric2:
                    st.metric(
                        label="P(NORMAL)",
                        value=f"{prob_normal * 100:.2f}%"
                    )

            # Step 3: Explainable AI Section
            st.divider()
            st.subheader("3. Explainable AI (XAI) – Visual Feature Attribution")
            st.markdown(
                "Explainability techniques highlight the anatomical regions and pathological patterns that drove the model's classification."
            )

            xai_tabs = st.tabs(["🔥 Grad-CAM Heatmap", "🟢 LIME Superpixels", "🔷 SHAP Feature Attribution"])

            # Tab 1: Grad-CAM
            with xai_tabs[0]:
                if model_info["supports_gradcam"]:
                    col_cam1, col_cam2, col_cam3 = st.columns([1, 1, 1])
                    
                    with st.spinner("Computing gradient activations on target convolutional layer..."):
                        try:
                            heatmap = generate_gradcam_heatmap(
                                model,
                                batch_tensor,
                                last_conv_layer_name=model_info["gradcam_layer"]
                            )
                            overlay = create_gradcam_overlay(display_arr, heatmap, alpha=0.50, colormap_name="jet")

                            with col_cam1:
                                st.image(display_arr, caption="Original Input Radiograph", use_container_width=True)
                            with col_cam2:
                                st.image(heatmap, caption="Grad-CAM Activation Map (Normalized)", use_container_width=True, clamp=True)
                            with col_cam3:
                                st.image(overlay, caption="Grad-CAM Overlay (Jet Colormap)", use_container_width=True)

                            st.success(
                                "💡 **Clinical Interpretation:** Warmer regions (red and yellow) denote lung parenchymal zones that contributed most positively to the prediction. In pneumonia cases, these typically correspond to focal consolidations, airspace opacities, or peribronchial thickening."
                            )
                        except Exception as e:
                            st.warning(f"Grad-CAM computation encountered an issue: {str(e)}")
                else:
                    st.info(
                        f"Grad-CAM is directly supported for Custom CNN. For {selected_model_name}, please use the LIME and SHAP explainability tabs below, which provide verified model-agnostic feature attributions."
                    )

            # Tab 2: LIME
            with xai_tabs[1]:
                st.markdown(
                    "**LIME (Local Interpretable Model-agnostic Explanations)** segments the radiograph into SLIC superpixels and measures their marginal contribution to the pneumonia prediction by perturbing individual regions."
                )
                
                col_l1, col_l2 = st.columns([1, 2])
                with col_l1:
                    lime_samples = st.slider("LIME Perturbation Samples", min_value=100, max_value=800, value=300, step=50)
                    lime_segments = st.slider("Superpixel Count (SLIC)", min_value=30, max_value=90, value=60, step=10)
                    run_lime_btn = st.button("Generate LIME Explanation", type="primary", key="btn_lime")
                
                with col_l2:
                    if run_lime_btn:
                        with st.spinner(f"Computing LIME superpixels ({lime_samples} perturbation samples)..."):
                            try:
                                wmap, bound_img, _ = run_lime_explanation(
                                    model,
                                    display_arr,
                                    num_samples=lime_samples,
                                    n_segments=lime_segments
                                )
                                c_sub1, c_sub2 = st.columns(2)
                                with c_sub1:
                                    st.image(bound_img, caption="Top Supporting Superpixels (Green Outline)", use_container_width=True)
                                with c_sub2:
                                    fig, ax = plt.subplots(figsize=(4, 4))
                                    fig.patch.set_facecolor('#0E1117')
                                    ax.set_facecolor('#1E222B')
                                    lim = max(float(np.abs(wmap).max()), 1e-6)
                                    im = ax.imshow(wmap, cmap="RdYlGn_r", vmin=-lim, vmax=lim)
                                    ax.axis("off")
                                    ax.set_title("Superpixel Weights (Red = Pneumonia)", color="#ECEFF1", fontsize=10)
                                    st.pyplot(fig, use_container_width=True)
                                    plt.close(fig)
                            except Exception as e:
                                st.error(f"LIME calculation failed: {str(e)}")
                    else:
                        st.info("Click 'Generate LIME Explanation' above to run superpixel attribution.")

            # Tab 3: SHAP
            with xai_tabs[2]:
                st.markdown(
                    "**SHAP (SHapley Additive exPlanations)** applies cooperative game theory with Gaussian blur masking to assign Shapley values to individual pixel groups."
                )
                col_s1, col_s2 = st.columns([1, 2])
                with col_s1:
                    shap_evals = st.slider("Max SHAP Evaluations", min_value=100, max_value=500, value=250, step=50)
                    run_shap_btn = st.button("Generate SHAP Explanation", type="primary", key="btn_shap")
                
                with col_s2:
                    if run_shap_btn:
                        with st.spinner(f"Running Partition SHAP ({shap_evals} evaluations)..."):
                            try:
                                sm = run_shap_explanation(model, display_arr, max_evals=shap_evals)
                                lim = max(float(np.abs(sm).max()), 1e-12)
                                
                                c_sh1, c_sh2 = st.columns(2)
                                with c_sh1:
                                    fig, ax = plt.subplots(figsize=(4, 4))
                                    fig.patch.set_facecolor('#0E1117')
                                    ax.imshow(sm, cmap="bwr", vmin=-lim, vmax=lim)
                                    ax.axis("off")
                                    ax.set_title("SHAP Attribution Map", color="#ECEFF1", fontsize=10)
                                    st.pyplot(fig, use_container_width=True)
                                    plt.close(fig)
                                with c_sh2:
                                    fig, ax = plt.subplots(figsize=(4, 4))
                                    fig.patch.set_facecolor('#0E1117')
                                    ax.imshow(display_arr)
                                    ax.imshow(sm, cmap="bwr", vmin=-lim, vmax=lim, alpha=0.55)
                                    ax.axis("off")
                                    ax.set_title("SHAP Overlay (Red = Supports Pneumonia)", color="#ECEFF1", fontsize=10)
                                    st.pyplot(fig, use_container_width=True)
                                    plt.close(fig)
                            except Exception as e:
                                st.error(f"SHAP calculation failed: {str(e)}")
                    else:
                        st.info("Click 'Generate SHAP Explanation' above to run Shapley value attribution.")

    with tab_performance:
        st.subheader("📊 Experimental Evaluation & Verified Model Benchmarks")
        st.markdown(
            """
            The models were rigorously evaluated on an independent test set of **904 chest radiographs** 
            (233 NORMAL, 671 PNEUMONIA) with patient-level stratification to ensure zero data leakage.
            """
        )

        # Verified test set table
        metrics_data = {
            "Metric": ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1-Score", "ROC-AUC", "PR-AUC"],
            "Custom CNN (Selected Best)": [
                f"{MODEL_CATALOG['Custom CNN']['metrics']['Accuracy']:.2%}",
                f"{MODEL_CATALOG['Custom CNN']['metrics']['Precision']:.2%}",
                f"{MODEL_CATALOG['Custom CNN']['metrics']['Recall (Sensitivity)']:.2%}",
                f"{MODEL_CATALOG['Custom CNN']['metrics']['Specificity']:.2%}",
                f"{MODEL_CATALOG['Custom CNN']['metrics']['F1-Score']:.2%}",
                f"{MODEL_CATALOG['Custom CNN']['metrics']['ROC-AUC']:.4f}",
                f"{MODEL_CATALOG['Custom CNN']['metrics']['PR-AUC']:.4f}",
            ],
            "DenseNet121 (Transfer Learning)": [
                f"{MODEL_CATALOG['DenseNet121']['metrics']['Accuracy']:.2%}",
                f"{MODEL_CATALOG['DenseNet121']['metrics']['Precision']:.2%}",
                f"{MODEL_CATALOG['DenseNet121']['metrics']['Recall (Sensitivity)']:.2%}",
                f"{MODEL_CATALOG['DenseNet121']['metrics']['Specificity']:.2%}",
                f"{MODEL_CATALOG['DenseNet121']['metrics']['F1-Score']:.2%}",
                f"{MODEL_CATALOG['DenseNet121']['metrics']['ROC-AUC']:.4f}",
                f"{MODEL_CATALOG['DenseNet121']['metrics']['PR-AUC']:.4f}",
            ]
        }
        st.table(metrics_data)

        st.markdown("##### Confusion Matrix Summary (Test Set, N = 904)")
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.markdown(
                """
                **Custom CNN:**
                - **True Positives (Pneumonia correctly identified):** 648
                - **True Negatives (Normal correctly identified):** 211
                - **False Positives (Normal misclassified as Pneumonia):** 22
                - **False Negatives (Pneumonia misclassified as Normal):** 23
                - *High sensitivity (96.57%) makes it ideal for triage screening.*
                """
            )
        with col_cm2:
            st.markdown(
                """
                **DenseNet121:**
                - **True Positives (Pneumonia correctly identified):** 629
                - **True Negatives (Normal correctly identified):** 223
                - **False Positives (Normal misclassified as Pneumonia):** 10
                - **False Negatives (Pneumonia misclassified as Normal):** 42
                - *High precision (98.44%) and specificity (95.71%) minimize false alarms.*
                """
            )

    with tab_about:
        st.subheader("📖 About PulmoLens & Clinical Methodology")
        st.markdown(
            """
            ### Background & Clinical Motivation
            Pneumonia is an acute respiratory infection affecting the alveoli and lung parenchyma. 
            Prompt and accurate radiological evaluation of chest X-rays (CXR) is vital for early intervention.
            However, interpreting subtle infiltrates and distinguishing viral from bacterial presentations 
            can pose significant diagnostic challenges, especially in resource-limited clinical environments.

            ### Dataset & Leakage Prevention
            - **Data Source:** Pediatric chest X-ray cohort from Guangzhou Women and Children's Medical Center.
            - **Audit for Patient Leakage:** Patient identifiers were extracted via regex (`personXX_`, `IM-XXXX`) to ensure that multiple radiographs from the same patient were strictly confined to either train, validation, or test sets.
            - **Splits:** 70% Train / 15% Validation / 15% Test (stratified by patient and label).

            ### Explainable AI in Medicine
            Deep learning models in medical imaging are often critiqued as "black boxes." PulmoLens addresses this 
            by embedding three complementary interpretability paradigms:
            1. **Grad-CAM:** Highlights high-gradient receptive fields in the final convolutional layer.
            2. **LIME:** Explains localized superpixel groupings through perturbation and local surrogates.
            3. **SHAP:** Uses cooperative game-theoretic Shapley values with blur masking.

            ### Deployment & Reliability
            - Pure inference architecture with zero external data dependencies.
            - Streamlit caching (`@st.cache_resource`) ensures near-instantaneous model loading.
            - Model file size (~15MB to ~40MB) fits comfortably within Git and cloud runtime constraints.
            """
        )


if __name__ == "__main__":
    main()

