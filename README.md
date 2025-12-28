# Early Alzheimer's Detection AI

## Hack4Health: AI for Alzheimer's Challenge

An AI model for early detection of Alzheimer's disease using brain MRI scans. This project uses transfer learning with EfficientNet-B0 to classify MRI images into four stages of dementia.

## Overview

### Problem Statement
Alzheimer's disease affects millions worldwide, and early detection is crucial for treatment planning and patient care. This project leverages deep learning to classify brain MRI scans into different stages of cognitive impairment.

### Classification Categories
- **NonDemented** - Healthy brain with no signs of dementia
- **VeryMildDemented** - Very early stage Alzheimer's
- **MildDemented** - Mild cognitive impairment
- **ModerateDemented** - Moderate stage Alzheimer's

## Key Features

- **Transfer Learning**: EfficientNet-B0 pre-trained on ImageNet
- **Data Augmentation**: Rotation, flipping, color jitter for robustness
- **Class Imbalance Handling**: Weighted loss function and weighted sampling
- **Model Interpretability**: Grad-CAM visualizations showing where the model looks
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1, AUC metrics

## Project Structure

```
early-alzheimers-detection-ai/
├── data/                           # Dataset (gitignored)
│   └── Alzheimer_MRI_4_classes_dataset/
│       ├── MildDemented/
│       ├── ModerateDemented/
│       ├── NonDemented/
│       └── VeryMildDemented/
├── notebooks/
│   └── alzheimers_mri_classification.ipynb  # Main reproducible notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Dataset loading and preprocessing
│   ├── model.py                    # Model architectures
│   ├── train.py                    # Training utilities
│   └── evaluate.py                 # Evaluation and visualization
├── checkpoints/                    # Saved models and outputs
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Jupyter Notebook (Recommended)
Open and run `notebooks/alzheimers_mri_classification.ipynb` - this contains the complete pipeline with documentation.

### Option 2: Google Colab
1. Upload the notebook to Google Colab
2. Upload your dataset to Colab or mount Google Drive
3. Update the `DATA_DIR` path in the Config class
4. Run all cells

## Dataset

The model is trained on the Alzheimer's MRI 4-Classes Dataset:
- **Total Images**: ~6,400
- **Classes**: 4 (with significant class imbalance)
- **Image Type**: Brain MRI slices (grayscale converted to RGB)

### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| NonDemented | 3,200 | 50% |
| VeryMildDemented | 2,240 | 35% |
| MildDemented | 896 | 14% |
| ModerateDemented | 64 | 1% |

## Model Architecture

**EfficientNet-B0** with custom classification head:
- Pre-trained backbone on ImageNet
- Custom classifier: Dropout → Linear(1280, 512) → ReLU → Dropout → Linear(512, 4)
- Total parameters: ~4.0M

## Results

After training, the model achieves:
- High accuracy on the test set
- Grad-CAM visualizations show the model focuses on relevant brain regions
- Detailed per-class metrics in the notebook

## Deliverables

1. **Jupyter Notebook**: [`notebooks/alzheimers_mri_classification.ipynb`](notebooks/alzheimers_mri_classification.ipynb)
   - Fully reproducible and documented
   - Works in Google Colab

2. **PDF Report**: [`REPORT.md`](REPORT.md)
   - Problem framing and motivation
   - Methods and data description
   - Evaluation metrics and discussion

3. **Model Card**: [`MODEL_CARD.md`](MODEL_CARD.md)
   - Model details and intended use
   - Bias, limitations, and fairness
   - Ethical considerations

4. **Visualizations** (generated when running notebook):
   - Class distribution plots
   - Training history (loss, accuracy, learning rate)
   - Confusion matrix
   - Grad-CAM heatmaps
   - Sample predictions

## Limitations

1. **Class Imbalance**: ModerateDemented has very few samples
2. **2D Only**: Model analyzes individual slices, not full 3D volumes
3. **Dataset Specificity**: May not generalize to different MRI scanners/protocols
4. **Not for Clinical Use**: Research/educational purposes only

## Future Improvements

1. Collect more data for underrepresented classes
2. Experiment with 3D CNNs for volumetric analysis
3. Implement ensemble methods
4. Add uncertainty quantification
5. Validate on external datasets

## Attribution

Developed for **Hack4Health: AI for Alzheimer's Challenge**

---

**Disclaimer**: This model is for research and educational purposes only. It should NOT be used for clinical diagnosis without proper validation and expert oversight.
