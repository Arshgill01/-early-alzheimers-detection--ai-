# Model Card: Alzheimer's MRI Classification

## Model Details

### Basic Information
| Attribute | Value |
|-----------|-------|
| **Model Name** | AlzheimerEfficientNet |
| **Version** | 1.0 |
| **Type** | Image Classification (CNN) |
| **Architecture** | EfficientNet-B0 + Custom Classifier |
| **Framework** | PyTorch |
| **Parameters** | ~4.0M total |
| **License** | Research/Educational Use |

### Model Description
A convolutional neural network for classifying brain MRI scans into four stages of Alzheimer's disease progression. Uses transfer learning from ImageNet-pretrained EfficientNet-B0 with a custom classification head.

### Developed By
Hack4Health: AI for Alzheimer's Challenge Participant

---

## Intended Use

### Primary Use Cases
- **Research Tool**: Exploratory analysis of brain MRI datasets
- **Educational**: Understanding AI applications in medical imaging
- **Screening Aid**: Preliminary flagging for radiologist review (NOT diagnosis)

### Out-of-Scope Uses
⚠️ **This model should NOT be used for:**
- Clinical diagnosis of Alzheimer's disease
- Making treatment decisions
- Replacing professional medical evaluation
- Insurance or legal determinations
- Any high-stakes medical decisions without expert oversight

---

## Training Data

### Dataset
**Alzheimer's MRI 4-Classes Dataset**
- **Total Samples**: 6,400 brain MRI slices
- **Image Format**: JPEG, grayscale converted to RGB
- **Resolution**: Resized to 224×224 pixels

### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| NonDemented | 3,200 | 50.0% |
| VeryMildDemented | 2,240 | 35.0% |
| MildDemented | 896 | 14.0% |
| ModerateDemented | 64 | 1.0% |

### Data Split
- Training: 70%
- Validation: 15%
- Test: 15%
- Stratified split maintaining class proportions

---

## Performance

### Metrics
| Metric | Expected Range |
|--------|----------------|
| Accuracy | 85-95% |
| Macro Precision | 0.70-0.85 |
| Macro Recall | 0.65-0.80 |
| Macro F1-Score | 0.70-0.85 |
| AUC (weighted) | >0.90 |

### Performance by Class
| Class | Expected Performance |
|-------|---------------------|
| NonDemented | High (large sample size) |
| VeryMildDemented | Good (adequate samples) |
| MildDemented | Moderate (limited samples) |
| ModerateDemented | Lower (only 64 samples) |

---

## Limitations

### Technical Limitations
1. **2D Analysis**: Analyzes individual MRI slices, not volumetric data
2. **Single Scanner**: Trained on one dataset; may not generalize across MRI protocols
3. **Class Imbalance**: Poor representation of ModerateDemented class (1%)
4. **Resolution**: Fixed 224×224 input may lose fine-grained details

### Dataset Limitations
- Demographics of source patients unknown
- Scanner manufacturer/settings not documented
- Slice selection methodology not specified
- No longitudinal validation data

---

## Bias and Fairness

### Known Biases
| Bias Type | Description | Mitigation |
|-----------|-------------|------------|
| **Class Imbalance** | ModerateDemented severely underrepresented | Weighted loss, oversampling |
| **Dataset Bias** | Single source dataset | Document limitation, recommend external validation |
| **Selection Bias** | Unknown patient selection criteria | Transparently report, avoid clinical claims |

### Fairness Considerations
- Model has not been evaluated across demographic groups (age, sex, ethnicity)
- Performance may vary for patients with comorbidities
- No differential performance analysis conducted

### Recommendations
- Validate on diverse, well-characterized cohorts before any clinical consideration
- Conduct fairness audits across demographic subgroups
- Use only as preliminary screening tool with expert oversight

---

## Ethical Considerations

### Privacy
- Model trained on publicly available research dataset
- No patient identifiers included in training/deployment

### Potential Harms
- **Misdiagnosis risk** if used without expert validation
- **Anxiety** from false positive predictions
- **Missed diagnosis** from false negatives, especially in minority classes
- **Over-reliance** on AI may reduce clinical scrutiny

### Responsible Use Guidelines
1. ✅ Use for research and educational purposes
2. ✅ Combine with expert radiologist review
3. ✅ Transparently communicate model limitations to stakeholders
4. ❌ Never use as sole basis for diagnosis
5. ❌ Do not deploy in clinical settings without rigorous validation

---

## Technical Specifications

### Input
- **Format**: RGB image (grayscale MRI converted to 3-channel)
- **Size**: 224 × 224 pixels
- **Normalization**: ImageNet mean/std

### Output
- **Format**: 4-class probability distribution
- **Classes**: NonDemented, VeryMildDemented, MildDemented, ModerateDemented

### Computational Requirements
- **Training**: GPU recommended (NVIDIA CUDA or Apple MPS)
- **Inference**: CPU capable (~100ms per image)
- **Memory**: ~2GB GPU memory for batch inference

---

## Interpretability

### Explainability Method
**Grad-CAM** (Gradient-weighted Class Activation Mapping)
- Visualizes regions the model attends to for predictions
- Enables verification that model focuses on clinically relevant areas
- Provides transparency for understanding model decisions

### Sample Explanations
The notebook generates Grad-CAM visualizations showing:
- Highlighted brain regions influencing classification
- Comparison across different dementia stages
- Verification of medically plausible attention patterns

---

## Citation

If you use this model, please cite:
```
Hack4Health: AI for Alzheimer's Challenge
Early Alzheimer's Detection from Brain MRI using Deep Learning
2024
```

---

## Contact

For questions or issues, refer to the project repository or contact the Hack4Health organizing team.

---

*Last Updated: December 2024*
