"""
Evaluation and visualization module for Alzheimer's MRI Classification.
Includes metrics computation, confusion matrix, and Grad-CAM explanations.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from tqdm import tqdm
import cv2

from .data_loader import IDX_TO_CLASS, CLASS_NAMES, denormalize


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get predictions from the model.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run on
        
    Returns:
        Tuple of (true_labels, predicted_labels, probabilities)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional, for AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
    }
    
    # Per-class metrics
    for i, class_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == i).astype(int)
        binary_pred = (y_pred == i).astype(int)
        metrics[f'precision_{class_name}'] = precision_score(binary_true, binary_pred, zero_division=0) * 100
        metrics[f'recall_{class_name}'] = recall_score(binary_true, binary_pred, zero_division=0) * 100
        metrics[f'f1_{class_name}'] = f1_score(binary_true, binary_pred, zero_division=0) * 100
    
    # AUC if probabilities provided
    if y_prob is not None:
        try:
            metrics['auc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro') * 100
        except ValueError:
            metrics['auc_macro'] = 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print metrics in a formatted way."""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print("\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.2f}%")
    print(f"  Precision (macro):  {metrics['precision_macro']:.2f}%")
    print(f"  Recall (macro):     {metrics['recall_macro']:.2f}%")
    print(f"  F1 Score (macro):   {metrics['f1_macro']:.2f}%")
    if 'auc_macro' in metrics:
        print(f"  AUC (macro):        {metrics['auc_macro']:.2f}%")
    
    print("\nPer-Class Metrics:")
    for class_name in CLASS_NAMES:
        print(f"\n  {class_name}:")
        print(f"    Precision: {metrics[f'precision_{class_name}']:.2f}%")
        print(f"    Recall:    {metrics[f'recall_{class_name}']:.2f}%")
        print(f"    F1 Score:  {metrics[f'f1_{class_name}']:.2f}%")
    
    print("\n" + "="*60)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
    
    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[0]
    )
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')
    
    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[1]
    )
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_class_distribution(
    labels: List[int],
    title: str = 'Class Distribution',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot class distribution.
    
    Args:
        labels: List of class labels
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    class_names = [IDX_TO_CLASS[u] for u in unique]
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(class_names)))
    
    bars = ax.bar(class_names, counts, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


class GradCAM:
    """
    Grad-CAM implementation for model interpretability.
    Generates visual explanations for CNN predictions.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: Trained CNN model
            target_layer: Target layer for Grad-CAM (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        
        # Weighted combination of activations
        cam = (weights * activations).sum(dim=0)  # (H, W)
        
        # ReLU
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def apply_gradcam(
    gradcam: GradCAM,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_class: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Grad-CAM to an image.
    
    Args:
        gradcam: GradCAM instance
        image_tensor: Preprocessed image tensor
        original_image: Original image as numpy array
        target_class: Target class (None for predicted)
        
    Returns:
        Tuple of (heatmap, overlay)
    """
    # Generate CAM
    cam = gradcam.generate(image_tensor, target_class)
    
    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = np.float32(heatmap) / 255 + np.float32(original_image)
    overlay = overlay / overlay.max()
    
    return heatmap, overlay


def plot_gradcam_samples(
    model: nn.Module,
    data_loader: DataLoader,
    target_layer: nn.Module,
    device: torch.device,
    num_samples: int = 8,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Plot Grad-CAM visualizations for sample images.
    
    Args:
        model: Trained model
        data_loader: DataLoader with test images
        target_layer: Target layer for Grad-CAM
        device: Device to run on
        num_samples: Number of samples to visualize
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    gradcam = GradCAM(model, target_layer)
    model.eval()
    
    # Get samples
    images_list = []
    labels_list = []
    
    for images, labels in data_loader:
        for img, lbl in zip(images, labels):
            images_list.append(img)
            labels_list.append(lbl.item())
            if len(images_list) >= num_samples:
                break
        if len(images_list) >= num_samples:
            break
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)
    
    for i, (img, true_label) in enumerate(zip(images_list, labels_list)):
        # Get prediction
        img_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred_label = output.argmax(dim=1).item()
        
        # Generate Grad-CAM
        cam = gradcam.generate(img_tensor, pred_label)
        
        # Denormalize image for display
        original = denormalize(img)
        
        # Resize CAM
        cam_resized = cv2.resize(cam, (original.shape[1], original.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Overlay
        overlay = 0.5 * original + 0.5 * heatmap
        overlay = overlay / overlay.max()
        
        # Plot
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f'Original\nTrue: {IDX_TO_CLASS[true_label]}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cam_resized, cmap='jet')
        axes[i, 1].set_title('Grad-CAM Heatmap')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(overlay)
        color = 'green' if pred_label == true_label else 'red'
        axes[i, 2].set_title(f'Overlay\nPred: {IDX_TO_CLASS[pred_label]}', color=color)
        axes[i, 2].axis('off')
    
    plt.suptitle('Grad-CAM Visualizations: Model Attention Areas', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_sample_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_samples: int = 16,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Plot sample predictions with original images.
    
    Args:
        model: Trained model
        data_loader: DataLoader with test images
        device: Device to run on
        num_samples: Number of samples
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    model.eval()
    
    images_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            
            for img, lbl, pred in zip(images, labels, preds):
                images_list.append(img.cpu())
                labels_list.append(lbl.item())
                preds_list.append(pred.item())
                
                if len(images_list) >= num_samples:
                    break
            if len(images_list) >= num_samples:
                break
    
    # Create grid
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (img, true_lbl, pred_lbl) in enumerate(zip(images_list, labels_list, preds_list)):
        original = denormalize(img)
        axes[i].imshow(original)
        
        true_name = IDX_TO_CLASS[true_lbl]
        pred_name = IDX_TO_CLASS[pred_lbl]
        
        color = 'green' if true_lbl == pred_lbl else 'red'
        axes[i].set_title(f'True: {true_name}\nPred: {pred_name}', fontsize=9, color=color)
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(images_list), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig
