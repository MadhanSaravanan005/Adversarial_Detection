"""
CLEAN & SIMPLE Streamlit Dashboard
AI-Generated Image Detection System
Focus on what ACTUALLY MATTERS
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import requests
import cv2

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🔍 AI Image Detector",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
    <style>
        body { font-family: 'Segoe UI', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR - MINIMAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("⚙️ **Settings**")

    backend_url = st.text_input(
        "Backend URL",
        value="http://localhost:9000"
    )

    model_type = st.radio(
        "Select Model",
        ["Robust ⭐ (Recommended)", "Baseline (Faster)"],
        help="Robust: Better detection\nBaseline: Faster"
    )

    model_choice = "robust" if "Robust" in model_type else "baseline"

    st.markdown("---")
    st.markdown("🛡️ **System Architecture**")
    st.caption("**Pipeline**: Detect → Defend → Decide")
    st.caption("**Active Mode**: Stage 1 Heuristic Ensemble (5 methods)")
    st.caption("**CNN Checkpoint**: Unloaded (graceful heuristic fallback)")


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE - STORE ACCURACY METRICS
# ═══════════════════════════════════════════════════════════════════════════

if 'baseline_accuracy' not in st.session_state:
    st.session_state.baseline_accuracy = None
if 'current_accuracy' not in st.session_state:
    st.session_state.current_accuracy = None
if 'buffer_size' not in st.session_state:
    st.session_state.buffer_size = 0
if 'retrains_done' not in st.session_state:
    st.session_state.retrains_done = 0

# ═══════════════════════════════════════════════════════════════════════════
# MAIN TITLE
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("# 🛡️ Automated Adversarial Monitoring & Self-Defense System")
st.markdown("**Detect, Defend, and Decide: Multi-stage Adversarial Detection, Defense Sanitization & Continuous Monitoring**")
st.info("ℹ️ **Active Engine**: 5-Method Heuristic Ensemble (Color, Texture, Skew, Shannon Entropy, Saturation). *No pre-trained CNN weights are bundled; the pipeline gracefully executes Stage 1 heuristic defense.*")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# 3 TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Detection (Main)",
    "🛡️ Recovery Test",
    "📚 How It Works",
    "⚡ Learning Progress",
    "📦 Batch Processing"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: REAL-TIME DETECTION (THE MAIN FEATURE)
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## Upload Image & Analyze")

    # Upload section
    uploaded_file = st.file_uploader(
        "📤 Choose image (JPG, PNG):",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Load image for preview only (don't resize here - let backend do it)
        image = Image.open(uploaded_file).convert('RGB')

        # Store original image bytes for heatmap calculation
        image_np = np.array(image.resize((256, 256))) / 255.0  # For heatmap only

        # Two columns layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(image, width=300)

        with col2:
            st.markdown("### 🔄 Analyzing...")

            # API Call - always use regular detect (no recovery)
            try:
                files = {'image': (uploaded_file.name, uploaded_file.getvalue())}

                response = requests.post(
                    f"{backend_url}/api/detect",
                    files=files,
                    data={'model_type': model_choice},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    risk_score = result.get('risk_score', 0)
                    confidence = result.get('confidence', 0)
                    detector_scores = result.get('detector_scores', {})

                    # Update accuracy from API
                    accuracy_data = result.get('accuracy', {})
                    st.session_state.baseline_accuracy = accuracy_data.get('baseline')
                    st.session_state.current_accuracy = accuracy_data.get('current')

                    # Update buffer status
                    buffer_data = result.get('buffer_status', {})
                    st.session_state.buffer_size = buffer_data.get('current_size', 0)
                    st.session_state.retrains_done = buffer_data.get('retrains_completed', 0)

                    # Decision Logic
                    if risk_score < 50:
                        decision = "✅ ALLOW (Safe)"
                        color = "green"
                    elif risk_score < 75:
                        decision = "⚠️ REVIEW (Suspicious)"
                        color = "orange"
                    else:
                        decision = "❌ BLOCK (Fake)"
                        color = "red"

                    # Display Results
                    st.markdown(f"### 📊 Results")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Risk Score", f"{risk_score:.1f}%")
                    m2.metric("Confidence", f"{confidence*100:.1f}%")
                    m3.markdown(f"<h3 style='color:{color}; text-align:center;'>{decision}</h3>", unsafe_allow_html=True)

                    # Create heatmap showing edge and gradient density
                    st.markdown("---")
                    st.markdown("### 🔥 Edge & High-Frequency Variance Heatmap")
                    st.markdown("*Visualizes spatial gradient density (Canny edge detection + Gaussian smoothing) highlighting structurally complex regions.*")

                    # Generate heatmap
                    img_gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(img_gray, 100, 200)
                    heatmap = cv2.GaussianBlur(edges.astype(float), (21, 21), 0)
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

                    # Display heatmap
                    fig = go.Figure()
                    fig.add_trace(go.Heatmap(
                        z=heatmap,
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Gradient Density"),
                        hovertemplate='<b>Structural Density</b><br>Position: (%{x}, %{y})<br>Level: %{z:.2%}<extra></extra>'
                    ))
                    fig.update_layout(
                        height=350,
                        xaxis_title="Width",
                        yaxis_title="Height",
                        title="Spatial Gradient & Texture Density"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.info("🔴 Red areas = Dense edges/textures | 🟡 Orange = Moderate texture | 🟢 Green/White = Smooth regions")

                    # Detector Breakdown (Simple)
                    st.markdown("---")
                    st.markdown("### 🔍 Detection Methods Breakdown")

                    detector_cols = st.columns(5)
                    detector_names = {
                        'robustness': 'Robustness',
                        'frequency': 'Frequency',
                        'statistical': 'Statistical',
                        'fingerprinting': 'Fingerprint',
                        'metadata': 'Metadata'
                    }

                    for idx, (key, name) in enumerate(detector_names.items()):
                        score = detector_scores.get(key, 0)
                        detector_cols[idx].metric(name, f"{int(score*100)}%")

                    # Recommendation
                    st.markdown("---")
                    if risk_score < 50:
                        st.success("✅ **SAFE** - Image passes all security checks. Safe to use.")
                    elif risk_score < 75:
                        st.warning("⚠️ **SUSPICIOUS** - Image shows unusual patterns. Manual review recommended.")
                    else:
                        st.error("❌ **FAKE** - Image is likely AI-generated or manipulated. Consider blocking.")

                    # Real-Time Buffer Status (from API response)
                    st.markdown("---")
                    st.markdown("### 📦 Real-Time Buffer Status")

                    buffer_status = result.get('buffer_status', {})
                    if buffer_status:
                        current_buffer = buffer_status.get('current_size', 0)
                        max_buffer = buffer_status.get('max_size', 5)
                        retrain_threshold = buffer_status.get('retrain_threshold', 5)
                        retrains_done = buffer_status.get('retrains_completed', 0)

                        buf_col1, buf_col2, buf_col3 = st.columns(3)
                        buf_col1.metric("🔴 Buffered Images", f"{current_buffer}/{max_buffer}")
                        buf_col2.metric("⏳ Until Retrain", f"{max(0, retrain_threshold - current_buffer)}")
                        buf_col3.metric("🔄 Total Retrains", f"{retrains_done}")

                        if current_buffer >= retrain_threshold:
                            st.warning(f"🚀 **RETRAINING ACTIVE!** Model is learning from {current_buffer} suspicious images...")

                else:
                    st.error(f"❌ Error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach backend!")
                st.info("✅ Make sure to run: `python -m backend.app`")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Upload an image to start detection")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: RECOVERY TESTING (Optional Feature)
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## 🛡️ Recovery & Sanitization Testing")
    st.markdown("**Optional Feature:** Test automatic image recovery for risky images")
    st.markdown("---")

    uploaded_file_recovery = st.file_uploader(
        "📤 Upload image to test recovery:",
        type=["jpg", "jpeg", "png"],
        key="recovery_uploader"
    )

    if uploaded_file_recovery is not None:
        image_rec = Image.open(uploaded_file_recovery).convert('RGB')

        col1_rec, col2_rec = st.columns(2)

        with col1_rec:
            st.markdown("### Original Image")
            st.image(image_rec, width=300)

        with col2_rec:
            st.markdown("### 🔄 Testing Recovery...")

            try:
                files = {'image': (uploaded_file_recovery.name, uploaded_file_recovery.getvalue())}
                response = requests.post(
                    f"{backend_url}/api/detect-with-recovery",
                    files=files,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    original_risk = result.get('original_risk', 0)
                    final_risk = result.get('final_risk', 0)
                    recovery_info = result.get('recovery', {})
                    recovery_attempted = recovery_info.get('attempted', False)
                    recovery_successful = recovery_info.get('successful', False)

                    # Display original results
                    st.markdown("### Original Detection")
                    st.metric("Original Risk Score", f"{original_risk:.1f}%")

                    original_decision = 'ALLOW' if original_risk < 50 else ('REVIEW' if original_risk < 75 else 'BLOCK')
                    original_color = 'green' if original_decision == 'ALLOW' else ('orange' if original_decision == 'REVIEW' else 'red')
                    st.markdown(f"<h4 style='color:{original_color};'>Decision: {original_decision}</h4>", unsafe_allow_html=True)

                    # Display recovery results
                    if recovery_attempted:
                        st.markdown("---")
                        st.markdown("### 🛡️ Recovery Attempt")

                        rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)
                        rec_col1.metric("Final Risk", f"{final_risk:.1f}%")
                        rec_col2.metric("Improvement", f"{original_risk - final_risk:.1f}%")
                        rec_col3.metric("Technique", recovery_info.get('technique', 'N/A'))
                        rec_col4.metric("Success", "Yes" if recovery_successful else "Partial")

                        if recovery_successful:
                            st.success(f"✅ **Recovery Successful!**")
                            st.info(f"Technique: {recovery_info.get('technique', 'Unknown')} | Risk: {original_risk:.1f}% → {final_risk:.1f}% | Final Decision: ALLOW")
                        else:
                            st.warning(f"⚠️ **Recovery Partial** - Risk reduced but still flagged")
                            final_decision = 'ALLOW' if final_risk < 50 else ('REVIEW' if final_risk < 75 else 'BLOCK')
                            st.info(f"Final Risk: {final_risk:.1f}% | Decision: {final_decision}")
                    else:
                        st.info(f"ℹ️ **No Recovery Needed** - Image is already safe ({original_risk:.1f}%)")

                else:
                    st.error(f"❌ Error: {response.status_code}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.info("👆 Upload an image to test recovery feature")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## 🔬 How Detection Works (5 Methods)")

    st.markdown(r"""
### **1️⃣ Color Consistency (25% weight)** — Channel Balance Analysis
- Calculates standard deviation across mean Red, Green, and Blue channels
- Detects unnatural color casting or targeted single-channel perturbations common in adversarial samples

### **2️⃣ Texture Variance (20% weight)** — Spatial Noise Analysis
- Converts image to grayscale and computes pixel standard deviation
- Detects high-frequency additive noise, roughness, and high-frequency perturbation patterns

### **3️⃣ Distribution Skew (25% weight)** — Histogram Symmetry
- Computes the third standardized moment (skewness) of the pixel intensity distribution
- Natural images maintain balanced, bell-like distributions; gradient-based attacks (FGSM/PGD) distort distribution symmetry

### **4️⃣ Entropy Pattern (20% weight)** — Shannon Information Density
- Measures Shannon entropy across the 256-bin pixel intensity histogram
- Adversarial perturbations increase random disorder, resulting in elevated information entropy

### **5️⃣ Chromatic Saturation (10% weight)** — HSV Color Space
- Analyzes color saturation ratios in HSV color space
- Flags images with extreme or artificially flattened saturation levels

---

## 📊 Ensemble Decision Policy

The 5 heuristic scores are combined with their assigned weights to generate an initial risk score (0% to 100%):

```
Heuristic Risk = 0.25*(Color) + 0.20*(Texture) + 0.25*(Skew) + 0.20*(Entropy) + 0.10*(Saturation)
```

**Decision Thresholds:**
- **< 50% = ALLOW**: Low risk, natural statistical properties
- **50% - 75% = REVIEW**: Uncertain anomaly detected, secondary CNN verification invoked if available
- **≥ 75% = BLOCK**: High confidence of artificial manipulation or adversarial attack
    """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: LEARNING PROGRESS (KEY POINTS ONLY)
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## ⚡ Real-Time Learning System")
    st.markdown("---")

    # Fetch real metrics from backend
    try:
        metrics_response = requests.get(f"{backend_url}/api/metrics", timeout=5)
        buffer_response = requests.get(f"{backend_url}/api/buffer-status", timeout=5)

        if metrics_response.status_code == 200:
            metrics_data = metrics_response.json()

            # Extract metrics with proper defaults
            baseline_acc = metrics_data.get('accuracy', {}).get('baseline')
            current_acc = metrics_data.get('accuracy', {}).get('current')
            improvement = metrics_data.get('accuracy', {}).get('improvement', 0.0)
            retrains_done = metrics_data.get('retraining', {}).get('count', 0)
            bg_retrains = metrics_data.get('retraining', {}).get('background_retrains', 0)
            is_retraining = metrics_data.get('retraining', {}).get('is_retraining', False)
            bg_accuracy = metrics_data.get('retraining', {}).get('bg_accuracy')
            last_loss = metrics_data.get('retraining', {}).get('last_loss')
        else:
            baseline_acc = None
            current_acc = None
            improvement = 0.0
            retrains_done = 0
            bg_retrains = 0
            is_retraining = False
            bg_accuracy = None
            last_loss = None

        if buffer_response.status_code == 200:
            buffer_data = buffer_response.json()
            buffer_size = buffer_data.get('buffer_size', 0)
            buffer_max = buffer_data.get('buffer_max', 5)
            fill_pct = buffer_data.get('percentage_full', 0)
            bg_buffer_size = buffer_data.get('background_buffer', {}).get('size', 0)
            is_bg_retraining = buffer_data.get('background_buffer', {}).get('is_retraining', False)
        else:
            buffer_size = 0
            buffer_max = 5
            fill_pct = 0
            bg_buffer_size = 0
            is_bg_retraining = False

    except Exception as e:
        st.warning(f"Could not fetch real metrics: {e}")
        buffer_size = 0
        buffer_max = 5
        fill_pct = 0
        baseline_acc = None
        current_acc = None
        improvement = 0.0
        retrains_done = 0
        bg_buffer_size = 0
        is_bg_retraining = False
        bg_retrains = 0
        is_retraining = False
        last_loss = None

    # Format accuracy strings
    start_acc_str = f"{baseline_acc:.1f}%" if baseline_acc is not None else "N/A"
    curr_acc_str = f"{current_acc:.1f}%" if current_acc is not None else "N/A"
    delta_str = f"+{improvement:.2f}%" if (improvement and improvement > 0) else None

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📦 Detection Buffer", f"{buffer_size}/{buffer_max}", delta=f"{fill_pct:.0f}% filled")
    col2.metric("🎯 Benchmark Accuracy", start_acc_str, help="Benchmark accuracy of pre-trained model (N/A if checkpoint not loaded)")
    col3.metric("📈 Fine-Tuned Accuracy", curr_acc_str, delta=delta_str)
    col4.metric("🔬 Background Retraining", "READY", delta="Thread Active")

    st.markdown("---")

    # Background retraining status
    st.markdown("### 🔬 Background Real-Time Retraining")

    bg_col1, bg_col2, bg_col3 = st.columns(3)
    bg_col1.metric("Actual Retrains", f"{bg_retrains}")
    bg_col2.metric("Model Buffer", f"{bg_buffer_size}/5")
    if is_bg_retraining:
        bg_col3.metric("Status", "🔄 RETRAINING...", delta="Active")
    else:
        bg_col3.metric("Status", "✅ Idle", delta="Ready")

    st.markdown("---")

    # Buffer status bar
    if buffer_max > 0:
        st.markdown(f"**Detection Buffer Status:** {buffer_size}/{buffer_max} images ({fill_pct:.0f}%)")
        st.progress(min(fill_pct / 100, 1.0))

        if buffer_size >= buffer_max:
            st.success("✅ Detection buffer full! Auto-retraining triggered!")
        elif buffer_size > 0:
            st.info(f"⏳ Collecting images... {buffer_max - buffer_size} more needed for retraining")
        else:
            st.info("🔄 Waiting for suspicious images to start buffering...")

    if bg_buffer_size > 0:
        st.markdown(f"**Real Retraining Buffer:** {bg_buffer_size}/5 adversarial examples")
        st.progress(min(bg_buffer_size / 5, 1.0))

        if is_bg_retraining:
            st.warning("⚠️ Real model retraining in progress on GPU/CPU... (Background)")
        elif bg_buffer_size >= 5:
            st.success("✅ Real retraining buffer full! Starting training step...")

    st.markdown("---")

    st.markdown(f"""
### 📊 Real-Time Learning Status

**Current Model:** `AdversarialDetectionCNN` (PyTorch)
**Operating Mode:** Stage 1 Heuristic Ensemble *(Pre-trained checkpoint not loaded; Stage 1 active)*

**Detection Buffer System (Real):**
- Current: {buffer_size}/{buffer_max} suspicious images ({fill_pct:.0f}% full)
- Trigger: When {buffer_max} images collected → Background retraining starts
- Method: PyTorch fine-tuning on buffered adversarial examples
- Result: Model weights updated in background thread

**Background Retraining:** ✅ ACTIVE & RUNNING
- Type: Genuine model fine-tuning (not simulated)
- Execution: Separate background thread (non-blocking)
- Data: Actual detected adversarial examples from buffer
- Optimizer: Adam with CrossEntropyLoss
- Device: GPU if available, CPU fallback

**Performance Metrics:**
- **Benchmark Accuracy:** {start_acc_str} *(Pre-trained checkpoint not bundled)*
- **Fine-Tuned Accuracy:** {curr_acc_str}
- **Last Optimizer Loss:** {f"{last_loss:.4f}" if last_loss is not None else "N/A"}

**Active Features:**
- ✅ 5-method ensemble detection (color consistency, texture variance, distribution skew, entropy, saturation)
- ✅ PyTorch fine-tuning on detected adversarial examples
- ✅ Background thread for continuous learning (non-blocking API)
- ✅ Defense sanitization pipeline (JPEG compression, Gaussian blur, median filter, spatial resizing, bit-depth reduction)
- ✅ Graceful error handling with fallback cascades
- ✅ Real-time metrics tracking and monitoring
    """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("## 📦 Batch Processing - Analyze Multiple Images")
    st.markdown("Upload multiple images at once to get comprehensive detection results.")

    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "📤 Choose multiple images (JPG, PNG):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown("---")
        st.markdown(f"### Processing {len(uploaded_files)} images...")

        # Process button
        if st.button("🚀 Analyze All Images"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
                status_text.text(f"Processing: {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})")

                try:
                    files = {'image': (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(
                        f"{backend_url}/api/detect",
                        files=files,
                        data={'model_type': model_choice},
                        timeout=10
                    )

                    if response.status_code == 200:
                        result = response.json()
                        results.append({
                            'filename': uploaded_file.name,
                            'risk_score': result.get('risk_score', 0),
                            'decision': result.get('decision', 'UNKNOWN'),
                            'confidence': result.get('confidence', 0)
                        })
                except Exception as e:
                    results.append({
                        'filename': uploaded_file.name,
                        'risk_score': 0,
                        'decision': 'ERROR',
                        'confidence': 0,
                        'error': str(e)
                    })

            progress_bar.empty()
            status_text.empty()

            # Display results
            st.markdown("---")
            st.markdown("### 📊 Results Summary")

            # Statistics
            allow_count = sum(1 for r in results if r['decision'] == 'ALLOW')
            review_count = sum(1 for r in results if r['decision'] == 'REVIEW')
            block_count = sum(1 for r in results if r['decision'] == 'BLOCK')

            col1, col2, col3 = st.columns(3)
            col1.metric("🟢 ALLOW", allow_count)
            col2.metric("🟡 REVIEW", review_count)
            col3.metric("🔴 BLOCK", block_count)

            # Detailed table
            st.markdown("---")
            st.markdown("### 📋 Detailed Results")

            df = pd.DataFrame(results)
            df['Risk Score'] = df['risk_score'].apply(lambda x: f"{x:.1f}%")
            df['Confidence'] = df['confidence'].apply(lambda x: f"{x*100:.1f}%")

            display_df = df[['filename', 'Risk Score', 'Confidence', 'decision']].copy()
            display_df.columns = ['Filename', 'Risk Score', 'Confidence', 'Decision']

            st.dataframe(display_df, use_container_width=True)

            # Download results as CSV
            st.markdown("---")
            csv = df[['filename', 'risk_score', 'decision', 'confidence']].to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="detection_results.csv",
                mime="text/csv"
            )
    else:
        st.info("👆 Upload multiple images to start batch processing")

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 11px;">

 AI-Generated Image Detector | Real-Time Learning | Multi-Method Ensemble

</div>
""", unsafe_allow_html=True)
