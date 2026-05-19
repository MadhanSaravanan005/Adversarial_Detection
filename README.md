# Adversarial Attack Detection & Defense System

A hybrid deep learning system for detecting and defending against adversarial attacks on image classification models using CIFAR-10 dataset.

## 🎯 Project Overview

This project implements a **multi-layer defense system** against adversarial attacks - specially crafted images designed to fool AI models while appearing normal to humans.

**Core Problem:** Hackers can create imperceptible modifications to images that cause AI models to misclassify them.

**Our Solution:** A three-stage detection pipeline that catches these attacks before they cause harm.

---

## 🏗️ System Architecture

### **Three-Stage Detection Pipeline**

```
STAGE 1: FAST HEURISTIC CHECK (5 methods, <50ms)
    ├─ Color Consistency Analysis (25%)
    ├─ Texture Variance Detection (20%)
    ├─ Distribution Skew Measurement (25%)
    ├─ Entropy Pattern Analysis (20%)
    └─ Chromatic Saturation Check (10%)
         ↓
STAGE 2: CNN VERIFICATION (Only if uncertain)
    ├─ Preprocessing (JPEG, Gaussian Blur, Median Filter)
    └─ Deep Learning Model Validation
         ↓
STAGE 3: FINAL DECISION
    ├─ Low Risk (< 50): ALLOW
    ├─ Medium Risk (50-75): REVIEW  
    └─ High Risk (≥ 75): BLOCK
```

---

## 📁 Project Structure

```
.
├── backend/                    # Core detection system
│   ├── app.py                 # Flask API server
│   ├── hybrid_detector.py     # Main detection logic
│   ├── hybrid_detection_with_recovery.py
│   ├── background_trainer.py
│   └── __init__.py
├── frontend/                  # Dashboard UI
│   ├── dashboard.py           # Streamlit dashboard
│   └── START_FRONTEND.bat
├── models/                    # Pre-trained models
│   ├── baseline_model.pth
│   ├── robust_model.pth
│   └── models_metadata.json
├── production_system/         # Production deployment
├── tools/                     # Utility scripts
├── requirements.txt           # Python dependencies
├── adversarial_detection_defense_cifar10.ipynb
└── project4.ipynb
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- TensorFlow/Keras
- Flask
- Streamlit

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ma

# Install dependencies
pip install -r requirements.txt
```

### Running the System

**Option 1: Start Backend API**
```bash
python backend/app.py
```

**Option 2: Start Frontend Dashboard**
```bash
python frontend/dashboard.py
```

**Option 3: Run Full System**
```bash
START_BACKEND.bat   # Terminal 1
START_FRONTEND.bat  # Terminal 2
```

---

## 🔬 Key Features

✅ **Hybrid Detection**: Combines fast heuristics + deep learning  
✅ **Real-time Processing**: < 50ms response time for Stage 1  
✅ **Multi-layer Defense**: 3-stage verification pipeline  
✅ **CIFAR-10 Optimized**: Specialized for small image classification  
✅ **Recovery Mechanism**: Recovers from uncertain detections  
✅ **Production Ready**: Containerized deployment support  

---

## 📊 Performance Metrics

- **Detection Accuracy**: Baseline & Robust models included
- **Latency**: < 100ms end-to-end
- **False Positive Rate**: Optimized across 5 heuristic methods
- **Preprocessing Effectiveness**: 3-layer defense (JPEG, Blur, Median)

---

## 📚 Documentation

See **METHODOLOGY.md** for detailed technical architecture, algorithm explanations, and research methodology.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, PyTorch
- **Frontend**: Streamlit
- **Models**: Deep Neural Networks (CNN)
- **Deployment**: Production System
- **Data**: CIFAR-10

---
