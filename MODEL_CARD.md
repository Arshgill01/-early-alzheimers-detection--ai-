# Model Card: Alzheimer's MRI Classification

> **Hackathon Submission**: [AI 4 Alzheimer's](https://ai4alzheimers.devpost.com/) | Hack4Health  
> **Last Updated**: December 2025

---

## Model Overview

| Attribute | Details |
|-----------|---------|
| **Model Name** | AlzheimerEfficientNet |
| **Version** | 1.0.0 |
| **Task** | Multi-class Image Classification |
| **Architecture** | EfficientNet-B0 + Custom Classification Head |
| **Framework** | PyTorch ≥2.0 |
| **Parameters** | ~4.0M (all trainable) |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Output** | 4-class probability distribution |
| **License** | Research & Educational Use Only |

### Description

A deep learning model for classifying brain MRI scans into **four stages of Alzheimer's disease progression**:

| Class | Description |
|-------|-------------|
| **NonDemented** | Healthy brain, no cognitive impairment |
| **VeryMildDemented** | Very early stage, subtle changes |
| **MildDemented** | Mild cognitive impairment (MCI) |
| **ModerateDemented** | Moderate stage Alzheimer's disease |

The model leverages **transfer learning** from ImageNet-pretrained EfficientNet-B0 with a custom classification head optimized for medical imaging.

---

## Intended Use

### ✅ Primary Use Cases

| Use Case | Description |
|----------|-------------|
| **Research Tool** | Exploratory analysis of brain MRI datasets |
| **Educational** | Understanding AI applications in medical imaging |
| **Screening Aid** | Preliminary flagging for radiologist review |
| **Benchmarking** | Baseline for Alzheimer's classification research |

### ⛔ Out-of-Scope Uses

This model should **NOT** be used for:

- ❌ Clinical diagnosis of Alzheimer's disease
- ❌ Making treatment decisions
- ❌ Replacing professional medical evaluation
- ❌ Insurance or legal determinations
- ❌ Any high-stakes medical decisions without expert oversight
- ❌ Production healthcare system deployment

---

## Training Data

### Dataset Information

| Attribute | Value |
|-----------|-------|
| **Dataset** | Alzheimer's MRI 4-Classes Dataset |
| **Total Samples** | 6,400 brain MRI slices |
| **Format** | JPEG (grayscale → RGB) |
| **Resolution** | Resized to 224 × 224 pixels |
| **Source** | [Hackathon Dataset](https://drive.google.com/drive/folders/1jGfWOHuA3kSbOQ4y26TI_ogBtDetw1SW) |

### Class Distribution

| Class | Samples | Percentage | Note |
|-------|---------|------------|------|
| NonDemented | 3,200 | 50.0% | Majority class |
| VeryMildDemented | 2,240 | 35.0% | Well-represented |
| MildDemented | 896 | 14.0% | Limited samples |
| ModerateDemented | 64 | 1.0% | ⚠️ Severely underrepresented |

### Data Split Strategy

| Split | Percentage | Samples | Stratification |
|-------|------------|---------|----------------|
| Training | 70% | ~4,480 | ✅ Maintained class ratios |
| Validation | 15% | ~960 | ✅ Maintained class ratios |
| Test | 15% | ~960 | ✅ Maintained class ratios |

---

## Training Methodology

### Architecture Details

```
Input (224 × 224 × 3)
         ↓
┌─────────────────────────────────┐
│   EfficientNet-B0 Backbone      │
│   (ImageNet Pre-trained)        │
│   - Compound Scaling            │
│   - MBConv Blocks               │
└─────────────────────────────────┘
         ↓
    Global Average Pooling
         ↓
    Dropout (p=0.3)
         ↓
    Linear (1280 → 512) + ReLU
         ↓
    Dropout (p=0.15)
         ↓
    Linear (512 → 4)
         ↓
    Softmax → Predictions
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Epochs | 20 (with early stopping) |
| Early Stopping Patience | 7 epochs |
| LR Scheduler | ReduceLROnPlateau (factor=0.5) |
| Loss Function | Cross-Entropy with class weights |

### Class Imbalance Mitigation

1. **Weighted Loss Function**: Inverse frequency class weights
2. **Weighted Random Sampling**: Oversample minority classes
3. **Data Augmentation**: 
   - Random horizontal flip (p=0.5)
   - Random rotation (±15°)
   - Random translation (±10%)
   - Color jitter (brightness/contrast ±20%)

---

## Performance Metrics

### Expected Performance (Test Set)

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| **Accuracy** | 85-95% | Overall classification correctness |
| **Precision (macro)** | 70-85% | Average across all classes |
| **Recall (macro)** | 65-80% | Sensitivity per class |
| **F1-Score (macro)** | 70-85% | Harmonic mean |
| **AUC-ROC (macro)** | >0.90 | Multi-class OvR |

### Per-Class Expected Performance

| Class | Expected Performance | Reasoning |
|-------|---------------------|-----------|
| NonDemented | ⭐⭐⭐⭐⭐ High | Large sample size (50%) |
| VeryMildDemented | ⭐⭐⭐⭐ Good | Adequate representation (35%) |
| MildDemented | ⭐⭐⭐ Moderate | Limited samples (14%) |
| ModerateDemented | ⭐⭐ Lower | Severely underrepresented (1%) |

---

## Limitations

### Technical Limitations

| Limitation | Description | Impact |
|------------|-------------|--------|
| **2D Analysis** | Analyzes individual slices, not 3D volumes | Limited anatomical context |
| **Fixed Resolution** | 224×224 input may lose fine details | Potential information loss |
| **Single Dataset** | Trained on one data source | Generalization concerns |
| **Class Imbalance** | ModerateDemented has only 64 samples | Poor minority class performance |

### Dataset Limitations

- Patient demographics unknown (age, sex, ethnicity)
- Scanner manufacturer/settings not documented
- Slice selection methodology not specified
- No longitudinal validation data available

---

## Bias and Fairness

### Known Biases

| Bias Type | Description | Mitigation |
|-----------|-------------|------------|
| **Class Imbalance** | ModerateDemented severely underrepresented | Weighted loss + oversampling |
| **Dataset Bias** | Single source dataset | Document + recommend external validation |
| **Selection Bias** | Unknown patient selection criteria | Transparent reporting |

### Fairness Considerations

- ⚠️ Not evaluated across demographic groups (age, sex, ethnicity)
- ⚠️ Performance may vary for patients with comorbidities
- ⚠️ No differential performance analysis conducted

### Recommendations for Responsible Use

1. Validate on diverse, well-characterized cohorts before any clinical consideration
2. Conduct fairness audits across demographic subgroups
3. Use only as a preliminary screening tool with expert oversight

---

## Ethical Considerations

### Privacy & Data Protection

- ✅ Model trained on publicly available research dataset
- ✅ No patient identifiers in training or deployment
- ✅ No personally identifiable information stored

### Potential Harms

| Risk | Description | Mitigation |
|------|-------------|------------|
| Misdiagnosis | False predictions without validation | Require expert review |
| Patient Anxiety | False positives causing distress | Clear communication of limitations |
| Missed Diagnosis | False negatives delaying treatment | Use as screening aid only |
| Over-reliance | Reduced clinical scrutiny | Emphasize human oversight |

### Responsible Use Guidelines

| ✅ DO | ❌ DON'T |
|-------|----------|
| Use for research and education | Use as sole diagnostic tool |
| Combine with expert radiologist review | Deploy without clinical validation |
| Communicate limitations transparently | Make treatment decisions solely on predictions |
| Validate on local data before use | Use in production healthcare systems |

---

## Model Interpretability

### Grad-CAM Visualization

The model includes **Gradient-weighted Class Activation Mapping (Grad-CAM)** for explainability:

- 🔍 Visualizes brain regions influencing predictions
- 🔍 Enables verification of clinically relevant attention patterns
- 🔍 Provides transparency for research and clinical review
- 🔍 Highlights areas of hippocampus, temporal lobe, and ventricles

---

## Technical Specifications

### Input Requirements

| Specification | Value |
|---------------|-------|
| Format | RGB image (3 channels) |
| Size | 224 × 224 pixels |
| Normalization | ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |

### Output Format

| Specification | Value |
|---------------|-------|
| Type | 4-dimensional probability vector |
| Range | [0, 1] (softmax normalized) |
| Classes | [MildDemented, ModerateDemented, NonDemented, VeryMildDemented] |

### Computational Requirements

| Resource | Requirement |
|----------|-------------|
| Training | GPU recommended (NVIDIA CUDA / Apple MPS) |
| Inference | CPU capable (~100ms/image) |
| GPU Memory | ~2GB for batch inference |
| Storage | ~16MB model checkpoint |

---

## Reproducibility

### Environment Setup

```bash
# Clone repository
git clone https://github.com/your-repo/early-alzheimers-detection-ai.git
cd early-alzheimers-detection-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Random Seeds

All experiments use fixed random seeds for reproducibility:
- Python random: 42
- NumPy: 42
- PyTorch: 42
- CUDA: 42

---

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{alzheimer_efficientnet_2025,
  title={Early Alzheimer's Detection from Brain MRI using Deep Learning},
  author={AI 4 Alzheimer's Hackathon Participant},
  year={2025},
  publisher={Hack4Health},
  note={AI 4 Alzheimer's Hackathon Submission}
}
```

---

## Acknowledgments

- **Hackathon**: [AI 4 Alzheimer's](https://ai4alzheimers.devpost.com/) by Hack4Health
- **Dataset**: Alzheimer's MRI 4-Classes Dataset
- **Framework**: PyTorch and torchvision

---

## Contact & Support

For questions or issues:
- 📧 Refer to the [project repository](https://github.com/your-repo/early-alzheimers-detection-ai)
- 🔗 Hackathon: [AI 4 Alzheimer's Discord](https://discord.com/invite/SZhaZcNh4D)

---

<p align="center">
  <em>⚠️ DISCLAIMER: This model is for research and educational purposes only. It is NOT a medical device and should NOT be used for clinical diagnosis.</em>
</p>
