# Alzheimer's Disease Detection from Brain MRI Scans
## Deep Learning Classification using Transfer Learning

**Hack4Health: AI for Alzheimer's Challenge**

---

## 1. Introduction

### Problem Statement
Alzheimer's disease (AD) is a progressive neurodegenerative disorder affecting over 55 million people worldwide, with numbers projected to triple by 2050. Early detection is critical for:
- Enabling timely intervention and treatment planning
- Allowing patients and families to prepare for disease progression
- Facilitating enrollment in clinical trials at optimal disease stages

Current diagnostic methods rely heavily on clinical assessments and expert radiologist interpretation of brain imaging, which can be subjective, time-consuming, and inaccessible in many regions. **This project develops an AI-assisted tool to classify brain MRI scans into four stages of cognitive impairment**, enabling faster, more consistent preliminary screening.

### Objective
Build a deep learning model to classify brain MRI scans into:
- **NonDemented**: Healthy brain, no signs of dementia
- **VeryMildDemented**: Very early stage cognitive impairment
- **MildDemented**: Mild cognitive impairment
- **ModerateDemented**: Moderate stage Alzheimer's disease

---

## 2. Data

### Dataset Description
We utilized the **Alzheimer's MRI 4-Classes Dataset**, containing 6,400 brain MRI slices organized into four classes representing disease progression stages.

| Class | Samples | Percentage |
|-------|---------|------------|
| NonDemented | 3,200 | 50.0% |
| VeryMildDemented | 2,240 | 35.0% |
| MildDemented | 896 | 14.0% |
| ModerateDemented | 64 | 1.0% |

### Class Imbalance Challenge
The dataset exhibits significant class imbalance, with ModerateDemented comprising only 1% of samples. We address this through:
1. **Weighted Loss Function**: Class-inverse frequency weights
2. **Weighted Random Sampling**: Oversampling minority classes during training
3. **Data Augmentation**: Generating synthetic variations of underrepresented classes

### Data Preprocessing
- **Image Resizing**: 224×224 pixels (standard for pretrained models)
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Data Split**: 70% training, 15% validation, 15% testing (stratified)

---

## 3. Methods

### Model Architecture
We employ **EfficientNet-B0** with transfer learning, chosen for its:
- **Efficiency**: Optimal accuracy-to-parameters ratio
- **Proven Performance**: State-of-the-art results on medical imaging tasks
- **Compound Scaling**: Balanced network depth, width, and resolution

**Architecture Configuration:**
```
EfficientNet-B0 Backbone (ImageNet pretrained)
    ↓
Global Average Pooling
    ↓
Dropout (0.3) → Linear (1280 → 512) → ReLU
    ↓
Dropout (0.15) → Linear (512 → 4) → Softmax
```

### Data Augmentation
Training augmentations to improve generalization:
- Random horizontal flip
- Random rotation (±15°)
- Random affine translation (±10%)
- Color jitter (brightness, contrast: ±20%)

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Epochs | 25 |
| Early Stopping | Patience = 7 |
| LR Scheduler | ReduceLROnPlateau |

### Model Interpretability
We implement **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize regions the model focuses on for predictions, enabling:
- Verification that the model attends to clinically relevant brain regions
- Transparency for potential clinical adoption
- Detection of spurious correlations in training

---

## 4. Evaluation

### Metrics
We evaluate performance using:
- **Accuracy**: Overall classification correctness
- **Precision**: Positive predictive value per class
- **Recall (Sensitivity)**: True positive rate per class
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve

### Expected Results
Based on the architecture and methodology, expected performance ranges:
- **Overall Accuracy**: 85-95%
- **Macro F1-Score**: 0.70-0.85
- **Per-class AUC**: >0.85 for majority classes

### Visualization Outputs
- Confusion matrix heatmap
- Training/validation loss and accuracy curves
- Grad-CAM attention visualizations
- Sample predictions with confidence scores

---

## 5. Discussion

### Strengths
1. **Transfer Learning**: Leverages ImageNet features for robust representations
2. **Interpretability**: Grad-CAM provides clinical transparency
3. **Class Imbalance Handling**: Multiple strategies for balanced learning
4. **Reproducibility**: Fully documented, Colab-compatible notebook

### Limitations
1. **2D Analysis Only**: Model analyzes individual slices, not 3D volumes
2. **Limited Moderate Class Data**: Only 64 samples may affect generalization
3. **Single Dataset**: May not generalize across different MRI scanners/protocols
4. **Research Tool Only**: Not validated for clinical diagnostic use

### Future Work
- Incorporate 3D CNNs for volumetric analysis
- Ensemble multiple architectures for robustness
- Validate on external datasets (ADNI, OASIS)
- Add uncertainty quantification for clinical confidence

---

## 6. Conclusion

This project demonstrates the feasibility of using deep learning for automated Alzheimer's disease stage classification from brain MRI scans. The EfficientNet-B0 transfer learning approach, combined with robust data augmentation and class imbalance handling, provides a foundation for AI-assisted preliminary screening. While not intended for clinical diagnosis, this work contributes to ongoing research in accessible, automated neuroimaging analysis.

---

**Repository**: Contains fully reproducible Jupyter notebook and modular Python source code.

**Disclaimer**: This model is for research and educational purposes only. It should NOT be used for clinical diagnosis without proper validation by medical professionals.
