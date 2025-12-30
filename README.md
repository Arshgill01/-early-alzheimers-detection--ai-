# 🧠 Early Alzheimer's Detection AI

<p align="center">
  <img src="https://img.shields.io/badge/AI%204%20Alzheimer's-Hackathon%20Submission-blue?style=for-the-badge" alt="Hackathon Badge">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-Educational-yellow?style=for-the-badge" alt="License">
</p>


<p align="center">
  <a href="https://ai4alzheimers.devpost.com/">Hackathon Page</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#results">Results</a> •
  <a href="#documentation">Documentation</a>
</p>

---

## 📋 Overview

An AI-powered model for **early detection of Alzheimer's disease** using brain MRI scans. This project uses **transfer learning with EfficientNet-B0** to classify MRI images into four stages of dementia progression.

### 🎯 Problem Statement

Alzheimer's disease affects **55+ million people worldwide** and early detection is crucial for treatment planning. This AI model assists in preliminary screening by classifying brain MRI scans into:

| Stage | Description |
|-------|-------------|
| 🟢 **NonDemented** | Healthy brain with no signs of dementia |
| 🟡 **VeryMildDemented** | Very early stage Alzheimer's |
| 🟠 **MildDemented** | Mild cognitive impairment |
| 🔴 **ModerateDemented** | Moderate stage Alzheimer's |

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔄 **Transfer Learning** | EfficientNet-B0 pre-trained on ImageNet |
| 📊 **Data Augmentation** | Rotation, flipping, color jitter for robustness |
| ⚖️ **Class Imbalance Handling** | Weighted loss function and weighted sampling |
| 🔍 **Model Interpretability** | Grad-CAM visualizations showing model attention |
| 📈 **Comprehensive Metrics** | Accuracy, Precision, Recall, F1, AUC-ROC |

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload `notebooks/alzheimers_mri_classification.ipynb` to Google Colab
2. Upload the dataset or mount Google Drive
3. Update `DATA_DIR` path in the Config class
4. Run all cells

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/early-alzheimers-detection-ai.git
cd early-alzheimers-detection-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/alzheimers_mri_classification.ipynb
```

---

## 📁 Project Structure

```
early-alzheimers-detection-ai/
│
├── 📓 notebooks/
│   └── alzheimers_mri_classification.ipynb  # Main reproducible notebook
│
├── 📦 src/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading & preprocessing
│   ├── model.py            # Model architectures
│   ├── train.py            # Training utilities
│   └── evaluate.py         # Evaluation & visualization
│
├── 📊 data/                 # Dataset (gitignored)
│   └── Alzheimer_MRI_4_classes_dataset/
│       ├── MildDemented/
│       ├── ModerateDemented/
│       ├── NonDemented/
│       └── VeryMildDemented/
│
├── 💾 checkpoints/          # Saved models & outputs
│
├── 📄 REPORT.md             # Technical report (2-3 pages)
├── 📋 MODEL_CARD.md         # Model documentation
├── 📝 requirements.txt      # Dependencies
└── 📖 README.md             # This file
```

---

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| **Total Images** | ~6,400 |
| **Classes** | 4 |
| **Format** | Brain MRI slices (grayscale → RGB) |
| **Source** | [Hackathon Dataset](https://drive.google.com/drive/folders/1jGfWOHuA3kSbOQ4y26TI_ogBtDetw1SW) |

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| NonDemented | 3,200 | 50% |
| VeryMildDemented | 2,240 | 35% |
| MildDemented | 896 | 14% |
| ModerateDemented | 64 | 1% |

> ⚠️ **Note**: Significant class imbalance exists. We address this with weighted sampling and loss functions.

---

## 🏗️ Model Architecture

**EfficientNet-B0** with custom classification head:

```
EfficientNet-B0 Backbone (ImageNet pretrained)
         ↓
   Global Average Pooling
         ↓
   Dropout (0.3) → Linear (1280 → 512) → ReLU
         ↓
   Dropout (0.15) → Linear (512 → 4) → Softmax
```

| Specification | Value |
|---------------|-------|
| **Total Parameters** | ~4.0M |
| **Input Size** | 224 × 224 × 3 |
| **Output** | 4-class probabilities |

---

## 📈 Results

### Expected Performance

| Metric | Expected Range |
|--------|----------------|
| **Accuracy** | 85-95% |
| **Precision (macro)** | 70-85% |
| **Recall (macro)** | 65-80% |
| **F1-Score (macro)** | 70-85% |
| **AUC-ROC** | >0.90 |

### Visualizations Generated

- ✅ Class distribution plots
- ✅ Training history (loss, accuracy, learning rate)
- ✅ Confusion matrix (counts & normalized)
- ✅ Grad-CAM attention heatmaps
- ✅ Sample predictions with confidence scores

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [REPORT.md](REPORT.md) | Technical report (problem, methods, evaluation) |
| [MODEL_CARD.md](MODEL_CARD.md) | Model details, limitations, ethical considerations |
| [Notebook](notebooks/alzheimers_mri_classification.ipynb) | Complete reproducible pipeline |

---

## ⚠️ Limitations

| Limitation | Description |
|------------|-------------|
| **Class Imbalance** | ModerateDemented has very few samples (1%) |
| **2D Analysis** | Individual slices, not full 3D brain volumes |
| **Dataset Specificity** | May not generalize across MRI scanners/protocols |
| **Research Only** | NOT validated for clinical diagnostic use |

---

## 🔮 Future Improvements

- [ ] Collect more data for underrepresented classes
- [ ] Experiment with 3D CNNs for volumetric analysis
- [ ] Implement ensemble methods
- [ ] Add uncertainty quantification
- [ ] Validate on external datasets (ADNI, OASIS)

---

## 📜 Hackathon Submission

### Deliverables

| Deliverable | Status | Link |
|-------------|--------|------|
| **PDF Report** (2-3 pages) | ✅ | [REPORT.md](REPORT.md) |
| **Reproducible Notebook** | ✅ | [Notebook](notebooks/alzheimers_mri_classification.ipynb) |
| **Model Card** | ✅ | [MODEL_CARD.md](MODEL_CARD.md) |

### Hackathon Links

- 🔗 [AI 4 Alzheimer's Hackathon](https://ai4alzheimers.devpost.com/)
- 💬 [Discord Community](https://discord.com/invite/SZhaZcNh4D)
- 📦 [Dataset](https://drive.google.com/drive/folders/1jGfWOHuA3kSbOQ4y26TI_ogBtDetw1SW)

---

## 🙏 Acknowledgments

- **Hackathon**: [AI 4 Alzheimer's](https://ai4alzheimers.devpost.com/) by Hack4Health
- **Dataset**: Alzheimer's MRI 4-Classes Dataset
- **Framework**: PyTorch, torchvision

---

## ⚖️ Disclaimer

<p align="center">
<strong>⚠️ IMPORTANT</strong>
</p>

> This model is for **research and educational purposes only**. It should **NOT** be used for clinical diagnosis without proper validation by qualified medical professionals. AI-assisted diagnosis should always be verified by healthcare experts.

---

<p align="center">
  Made with ❤️ for the <strong>AI 4 Alzheimer's Hackathon</strong>
</p>
